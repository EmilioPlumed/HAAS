"""Phase 2 experiment orchestration — NAS-Bench-201 search trajectory.

Runs HAAS and baseline agents in the NAS-Bench-201 cell search space and
evaluates each proposed cell against the benchmark. Logs:
  - Full trajectory (cell strings at each step, commitment scores)
  - NAS-Bench-201 accuracy for each proposed cell
  - Structural diversity metrics (Hamming distance, unique cells, compliance rate)
  - W&B trajectory table + summary scalars

W&B run names: phase2_{condition}_{hypothesis_id}
"""

import json
from typing import Optional

import wandb

from haas.agents.nas_committed_agent import NASCommittedAgent
from haas.agents.nas_baseline_agent import NASBaselineAgent
from haas.agents.nas_critic_agent import NASCriticAgent
from haas.agents.critic_agent import CriticAgent
from haas.config import ExperimentConfig, LLMConfig
from haas.hypotheses import Hypothesis, HYPOTHESIS_POOL
from haas.llm_client import LLMClient
from haas.logging_utils import init_run, finish_run
from haas.nas_bench import (
    validate_cell_string,
    canonicalize_cell_string,
    parse_cell_string,
    satisfies_constraint,
    hamming_distance,
    mean_pairwise_hamming,
    unique_cell_count,
    hypothesis_compliance_rate,
    load_api,
    query_accuracy,
    HYPOTHESIS_CONSTRAINTS,
)

# Phase 2 hypotheses — only those with NAS-Bench-201 constraints defined
PHASE2_HYPOTHESES = list(HYPOTHESIS_CONSTRAINTS.keys())


def _extract_cell_string(result: dict) -> Optional[str]:
    """Pull the cell_string field, canonicalize node indices, and validate.

    LLMs sometimes use the wrong ~N source-node index (e.g. none~0 instead of
    none~1). canonicalize_cell_string extracts the 6 op names in order and
    rebuilds the string with correct indices so the NAS-Bench-201 API can look
    it up.
    """
    cell_str = result.get("cell_string", "")
    return canonicalize_cell_string(cell_str)  # returns None if ops are invalid


def _log_nas_step(
    logger,
    step: int,
    phase: str,
    agent: str,
    event_type: str,
    result: dict,
    cell_str: Optional[str],
    acc: Optional[float],
    compliant: Optional[bool],
) -> None:
    """Log a trajectory event with NAS metadata appended to content."""
    content = json.dumps(result)
    nas_meta = {
        "cell_string": cell_str or "INVALID",
        "cifar10_test_acc": acc,
        "hypothesis_compliant": compliant,
    }
    content_with_meta = content + "\n[NAS]" + json.dumps(nas_meta)
    logger.log_event(step, phase, agent, event_type, content_with_meta)

    if logger._use_wandb:
        wandb.log({
            f"{agent}_cell": cell_str or "INVALID",
            f"{agent}_acc": acc if acc is not None else -1.0,
            f"{agent}_compliant": int(compliant) if compliant is not None else -1,
            "step": step,
        })


# ---------------------------------------------------------------------------
# HAAS condition
# ---------------------------------------------------------------------------

def run_haas_phase2(
    hypothesis: Hypothesis,
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    api=None,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Run NAS-Bench-201 HAAS experiment. Returns summary dict."""
    logger = init_run(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    client = LLMClient(llm_config)
    committed = NASCommittedAgent(hypothesis, client)
    critic = NASCriticAgent(client)

    print(f"\n{'='*60}")
    print(f"Phase 2 HAAS | hypothesis: {hypothesis.id}")
    print(f"Commitment steps: {exp_config.commitment_steps}")
    print(f"{'='*60}\n")

    all_cells: list[Optional[str]] = []
    committed_cells: list[Optional[str]] = []

    # --- Step 0: initial proposal ---
    proposal = committed.initial_proposal()
    cell_str = _extract_cell_string(proposal)

    compliant = None
    if cell_str and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
        try:
            ops = parse_cell_string(cell_str)
            compliant = satisfies_constraint(ops, hypothesis.id)
        except ValueError:
            compliant = False

    acc = query_accuracy(api, cell_str) if api and cell_str else None
    _log_nas_step(logger, 0, "commitment", "committed", "proposal", proposal, cell_str, acc, compliant)

    score_text = proposal.get("hypothesis_rationale", json.dumps(proposal))
    score = client.score_commitment(hypothesis.statement, score_text)
    logger.log_commitment_score(0, score)

    committed.record_acc(acc)
    all_cells.append(cell_str)
    committed_cells.append(cell_str)
    print(f"  Step 0 | cell: {cell_str} | acc: {acc} | compliant: {compliant} | score: {score}/10\n")

    # --- Commitment phase ---
    for step in range(1, exp_config.commitment_steps):
        current_cell = cell_str or json.dumps(proposal)
        challenge = critic.challenge(current_cell, hypothesis.id, hypothesis.statement, acc)
        logger.log_event(step, "commitment", "critic", "challenge", challenge)

        defense = committed.steelman_and_refine(challenge, acc)
        cell_str = _extract_cell_string(defense)

        compliant = None
        if cell_str and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
            try:
                ops = parse_cell_string(cell_str)
                compliant = satisfies_constraint(ops, hypothesis.id)
            except ValueError:
                compliant = False

        acc = query_accuracy(api, cell_str) if api and cell_str else None
        committed.record_acc(acc)
        _log_nas_step(logger, step, "commitment", "committed", "defense", defense, cell_str, acc, compliant)

        score_text = defense.get("defense", json.dumps(defense))
        score = client.score_commitment(hypothesis.statement, score_text)
        logger.log_commitment_score(step, score)

        all_cells.append(cell_str)
        committed_cells.append(cell_str)
        proposal = {**proposal, "hypothesis_rationale": defense.get("defense", "")}
        print(f"  Step {step} | cell: {cell_str} | acc: {acc} | compliant: {compliant} | score: {score}/10\n")

    # --- Final proposal ---
    final = committed.finalize()
    final_cell = _extract_cell_string(final)

    final_compliant = None
    if final_cell and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
        try:
            ops = parse_cell_string(final_cell)
            final_compliant = satisfies_constraint(ops, hypothesis.id)
        except ValueError:
            final_compliant = False

    final_acc = query_accuracy(api, final_cell) if api and final_cell else None
    _log_nas_step(
        logger, exp_config.commitment_steps, "evaluation", "committed", "final",
        final, final_cell, final_acc, final_compliant,
    )

    final_score = client.score_commitment(
        hypothesis.statement, final.get("design_rationale", json.dumps(final))
    )
    logger.log_commitment_score(exp_config.commitment_steps, final_score)
    all_cells.append(final_cell)
    committed_cells.append(final_cell)

    # --- Diversity and compliance metrics ---
    valid_cells = [c for c in committed_cells if c is not None]
    diversity = mean_pairwise_hamming(valid_cells)
    n_unique = unique_cell_count(valid_cells)
    compliance = hypothesis_compliance_rate(valid_cells, hypothesis.id) if hypothesis.id in HYPOTHESIS_CONSTRAINTS else None

    if logger._use_wandb:
        wandb.summary.update({
            "final_cell": final_cell or "INVALID",
            "final_cifar10_acc": final_acc,
            "final_compliant": final_compliant,
            "trajectory_diversity": diversity,
            "unique_cells": n_unique,
            "compliance_rate": compliance,
        })

    print(f"  Final cell: {final_cell}")
    print(f"  Final acc: {final_acc} | compliant: {final_compliant} | score: {final_score}/10")
    print(f"  Diversity: {diversity:.2f} | unique cells: {n_unique} | compliance: {compliance}")

    finish_run(logger)
    return {
        "condition": "haas",
        "hypothesis_id": hypothesis.id,
        "final_cell": final_cell,
        "final_accuracy": final_acc,
        "final_compliant": final_compliant,
        "final_commitment_score": final_score,
        "trajectory_cells": committed_cells,
        "trajectory_diversity": diversity,
        "unique_cells": n_unique,
        "compliance_rate": compliance,
    }


# ---------------------------------------------------------------------------
# Baseline condition
# ---------------------------------------------------------------------------

def run_baseline_phase2(
    hypothesis: Hypothesis,
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    api=None,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Run NAS-Bench-201 baseline experiment. Returns summary dict."""
    logger = init_run(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    client = LLMClient(llm_config)
    baseline = NASBaselineAgent(hypothesis, client)
    critic = CriticAgent(client)

    print(f"\n{'='*60}")
    print(f"Phase 2 Baseline | hypothesis topic: {hypothesis.id}")
    print(f"Refinement steps: {exp_config.commitment_steps}")
    print(f"{'='*60}\n")

    baseline_cells: list[Optional[str]] = []

    # --- Step 0 ---
    proposal = baseline.initial_proposal()
    cell_str = _extract_cell_string(proposal)

    compliant = None
    if cell_str and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
        try:
            ops = parse_cell_string(cell_str)
            compliant = satisfies_constraint(ops, hypothesis.id)
        except ValueError:
            compliant = False

    acc = query_accuracy(api, cell_str) if api and cell_str else None
    _log_nas_step(logger, 0, "refinement", "baseline", "proposal", proposal, cell_str, acc, compliant)

    score_text = proposal.get("hypothesis_rationale", json.dumps(proposal))
    score = client.score_commitment(hypothesis.statement, score_text)
    logger.log_commitment_score(0, score)

    baseline.record_acc(acc)
    baseline_cells.append(cell_str)
    print(f"  Step 0 | cell: {cell_str} | acc: {acc} | compliant: {compliant} | score: {score}/10\n")

    # --- Refinement phase ---
    for step in range(1, exp_config.commitment_steps):
        critique = critic.challenge(json.dumps(proposal))
        logger.log_event(step, "refinement", "critic", "challenge", critique)

        proposal = baseline.refine(critique, acc)
        cell_str = _extract_cell_string(proposal)

        compliant = None
        if cell_str and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
            try:
                ops = parse_cell_string(cell_str)
                compliant = satisfies_constraint(ops, hypothesis.id)
            except ValueError:
                compliant = False

        acc = query_accuracy(api, cell_str) if api and cell_str else None
        baseline.record_acc(acc)
        _log_nas_step(logger, step, "refinement", "baseline", "refinement", proposal, cell_str, acc, compliant)

        score_text = proposal.get("changes_made", proposal.get("defense", json.dumps(proposal)))
        score = client.score_commitment(hypothesis.statement, score_text)
        logger.log_commitment_score(step, score)

        baseline_cells.append(cell_str)
        print(f"  Step {step} | cell: {cell_str} | acc: {acc} | compliant: {compliant} | score: {score}/10\n")

    # --- Final proposal ---
    final = baseline.finalize()
    final_cell = _extract_cell_string(final)

    final_compliant = None
    if final_cell and hypothesis.id in HYPOTHESIS_CONSTRAINTS:
        try:
            ops = parse_cell_string(final_cell)
            final_compliant = satisfies_constraint(ops, hypothesis.id)
        except ValueError:
            final_compliant = False

    final_acc = query_accuracy(api, final_cell) if api and final_cell else None
    _log_nas_step(
        logger, exp_config.commitment_steps, "evaluation", "baseline", "final",
        final, final_cell, final_acc, final_compliant,
    )

    final_score = client.score_commitment(
        hypothesis.statement, final.get("design_rationale", json.dumps(final))
    )
    logger.log_commitment_score(exp_config.commitment_steps, final_score)
    baseline_cells.append(final_cell)

    # --- Diversity and compliance ---
    valid_cells = [c for c in baseline_cells if c is not None]
    diversity = mean_pairwise_hamming(valid_cells)
    n_unique = unique_cell_count(valid_cells)
    compliance = hypothesis_compliance_rate(valid_cells, hypothesis.id) if hypothesis.id in HYPOTHESIS_CONSTRAINTS else None

    if logger._use_wandb:
        wandb.summary.update({
            "final_cell": final_cell or "INVALID",
            "final_cifar10_acc": final_acc,
            "final_compliant": final_compliant,
            "trajectory_diversity": diversity,
            "unique_cells": n_unique,
            "compliance_rate": compliance,
        })

    print(f"  Final cell: {final_cell}")
    print(f"  Final acc: {final_acc} | compliant: {final_compliant} | score: {final_score}/10")
    print(f"  Diversity: {diversity:.2f} | unique cells: {n_unique} | compliance: {compliance}")

    finish_run(logger)
    return {
        "condition": "baseline",
        "hypothesis_id": hypothesis.id,
        "final_cell": final_cell,
        "final_accuracy": final_acc,
        "final_compliant": final_compliant,
        "final_commitment_score": final_score,
        "trajectory_cells": baseline_cells,
        "trajectory_diversity": diversity,
        "unique_cells": n_unique,
        "compliance_rate": compliance,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_phase2_experiment(
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    api=None,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Select hypothesis, load benchmark API if needed, dispatch to condition."""
    if exp_config.hypothesis_id is None:
        raise ValueError("Phase 2 requires an explicit hypothesis_id.")
    if exp_config.hypothesis_id not in PHASE2_HYPOTHESES:
        raise ValueError(
            f"Hypothesis {exp_config.hypothesis_id!r} has no NAS-Bench-201 constraint. "
            f"Phase 2 hypotheses: {PHASE2_HYPOTHESES}"
        )

    hypothesis = HYPOTHESIS_POOL[exp_config.hypothesis_id]

    # Load benchmark API if not provided
    if api is None:
        try:
            api = load_api()
            print(f"NAS-Bench-201 API loaded.")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Running without benchmark evaluation (cell strings will be proposed but not scored).")
            api = None

    if exp_config.condition == "haas":
        return run_haas_phase2(hypothesis, llm_config, exp_config, api, wandb_api_key, no_wandb)
    elif exp_config.condition == "baseline":
        return run_baseline_phase2(hypothesis, llm_config, exp_config, api, wandb_api_key, no_wandb)
    else:
        raise ValueError(f"Unknown condition {exp_config.condition!r}. Use 'haas' or 'baseline'.")
