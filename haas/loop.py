"""Experiment orchestration.

run_haas_experiment  — committed agent + critic loop
run_baseline_experiment — standard generate-critique-refine loop

Both functions follow the same external signature so run_experiment.py can
dispatch between them uniformly.
"""

import json
import random
from typing import Optional

from haas.agents.committed_agent import CommittedAgent
from haas.agents.baseline_agent import BaselineAgent
from haas.agents.critic_agent import CriticAgent
from haas.config import ExperimentConfig, LLMConfig
from haas.hypotheses import Hypothesis, HYPOTHESIS_POOL
from haas.llm_client import LLMClient
from haas.logging_utils import TrajectoryLogger, init_run, finish_run


# ---------------------------------------------------------------------------
# HAAS condition
# ---------------------------------------------------------------------------

def run_haas_experiment(
    hypothesis: Hypothesis,
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Run the committed-agent / critic loop and return a summary dict."""
    logger = init_run(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    client = LLMClient(llm_config)
    committed = CommittedAgent(hypothesis, client)
    critic = CriticAgent(client)

    print(f"\n{'='*60}")
    print(f"HAAS run | hypothesis: {hypothesis.id}")
    print(f"Commitment steps: {exp_config.commitment_steps}")
    print(f"{'='*60}\n")

    # --- Commitment phase ---
    proposal = committed.initial_proposal()
    logger.log_event(0, "commitment", "committed", "proposal", json.dumps(proposal))

    score_text = proposal.get("hypothesis_rationale", json.dumps(proposal))
    score = client.score_commitment(hypothesis.statement, score_text)
    logger.log_commitment_score(0, score)
    print(f"  Commitment score after step 0: {score}/10\n")

    for step in range(1, exp_config.commitment_steps):
        challenge = critic.challenge(json.dumps(proposal))
        logger.log_event(step, "commitment", "critic", "challenge", challenge)

        defense = committed.steelman_and_refine(challenge)
        logger.log_event(step, "commitment", "committed", "defense", json.dumps(defense))

        # Track whether the proposal changed
        refinement = defense.get("refinement_summary", "none")
        if refinement.strip().lower() != "none":
            # Merge the refinement back into the proposal dict for the next step
            proposal = {**proposal, "hypothesis_rationale": defense["defense"]}

        score_text = defense.get("defense", json.dumps(defense))
        score = client.score_commitment(hypothesis.statement, score_text)
        logger.log_commitment_score(step, score)
        print(f"  Commitment score after step {step}: {score}/10\n")

    # --- Evaluation phase ---
    final = committed.finalize()
    logger.log_event(
        exp_config.commitment_steps, "evaluation", "committed", "final", json.dumps(final)
    )
    final_score = client.score_commitment(
        hypothesis.statement, final.get("design_rationale", json.dumps(final))
    )
    logger.log_commitment_score(exp_config.commitment_steps, final_score)
    print(f"  Final commitment score: {final_score}/10\n")

    finish_run(logger)
    return {
        "condition": "haas",
        "hypothesis_id": hypothesis.id,
        "final_proposal": final,
        "final_commitment_score": final_score,
    }


# ---------------------------------------------------------------------------
# Baseline condition
# ---------------------------------------------------------------------------

def run_baseline_experiment(
    hypothesis: Hypothesis,
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Run the standard generate-critique-refine loop and return a summary dict."""
    logger = init_run(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    client = LLMClient(llm_config)
    baseline = BaselineAgent(hypothesis, client)
    critic = CriticAgent(client)

    print(f"\n{'='*60}")
    print(f"Baseline run | hypothesis topic: {hypothesis.id}")
    print(f"Refinement steps: {exp_config.commitment_steps}")
    print(f"{'='*60}\n")

    proposal = baseline.initial_proposal()
    logger.log_event(0, "refinement", "baseline", "proposal", json.dumps(proposal))

    score_text = proposal.get("hypothesis_rationale", json.dumps(proposal))
    score = client.score_commitment(hypothesis.statement, score_text)
    logger.log_commitment_score(0, score)
    print(f"  Alignment score after step 0: {score}/10\n")

    for step in range(1, exp_config.commitment_steps):
        critique = critic.challenge(json.dumps(proposal))
        logger.log_event(step, "refinement", "critic", "challenge", critique)

        proposal = baseline.refine(critique)
        logger.log_event(step, "refinement", "baseline", "refinement", json.dumps(proposal))

        score_text = proposal.get("changes_made", json.dumps(proposal))
        score = client.score_commitment(hypothesis.statement, score_text)
        logger.log_commitment_score(step, score)
        print(f"  Alignment score after step {step}: {score}/10\n")

    # Evaluation phase
    final = baseline.finalize()
    logger.log_event(
        exp_config.commitment_steps, "evaluation", "baseline", "final", json.dumps(final)
    )
    final_score = client.score_commitment(
        hypothesis.statement, final.get("design_rationale", json.dumps(final))
    )
    logger.log_commitment_score(exp_config.commitment_steps, final_score)
    print(f"  Final alignment score: {final_score}/10\n")

    finish_run(logger)
    return {
        "condition": "baseline",
        "hypothesis_id": hypothesis.id,
        "final_proposal": final,
        "final_commitment_score": final_score,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_experiment(
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> dict:
    """Select hypothesis, dispatch to the right condition, return summary."""
    if exp_config.hypothesis_id is None:
        hypothesis = random.choice(list(HYPOTHESIS_POOL.values()))
    else:
        hypothesis = HYPOTHESIS_POOL[exp_config.hypothesis_id]

    if exp_config.condition == "haas":
        return run_haas_experiment(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    elif exp_config.condition == "baseline":
        return run_baseline_experiment(hypothesis, llm_config, exp_config, wandb_api_key, no_wandb)
    else:
        raise ValueError(
            f"Unknown condition '{exp_config.condition}'. Use 'haas' or 'baseline'."
        )
