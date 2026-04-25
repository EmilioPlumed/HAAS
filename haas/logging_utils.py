"""Weights & Biases logging helpers for HAAS experiments.

Each experiment run logs:
  - A trajectory Table: one row per event (proposal / challenge / defense / etc.)
  - Summary scalars: commitment scores at each step, final score
  - Config: hypothesis, condition, model
"""

from typing import Any, Dict, List, Optional
import json
import os
import wandb

from haas.hypotheses import Hypothesis
from haas.config import ExperimentConfig, LLMConfig


class TrajectoryLogger:
    """Accumulates trajectory events and flushes them to W&B at run end."""

    COLUMNS = ["step", "phase", "agent", "event_type", "content"]

    def __init__(self, use_wandb: bool = True, local_path: Optional[str] = None):
        self._rows: List[List[Any]] = []
        self._commitment_scores: Dict[int, int] = {}  # step -> score
        self._use_wandb = use_wandb
        self._local_path = local_path  # JSONL file for full-fidelity backup

    def log_event(
        self,
        step: int,
        phase: str,
        agent: str,
        event_type: str,
        content: str,
    ) -> None:
        """Record one trajectory event.

        Args:
            step: Reasoning step index (0-based)
            phase: "commitment" or "evaluation"
            agent: "committed" | "baseline" | "critic"
            event_type: "proposal" | "challenge" | "defense" | "refinement" | "final"
            content: Full text of the message
        """
        self._rows.append([step, phase, agent, event_type, content])
        if self._local_path:
            os.makedirs(os.path.dirname(self._local_path), exist_ok=True)
            with open(self._local_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "phase": phase, "agent": agent,
                                    "event_type": event_type, "content": content}) + "\n")
        # Mirror to stdout for live monitoring
        print(f"  [{phase:10s}] step={step} {agent:10s} / {event_type:12s}")
        if len(content) > 120:
            print(f"    {content[:117]}...")
        else:
            print(f"    {content}")

    def log_commitment_score(self, step: int, score: int) -> None:
        self._commitment_scores[step] = score
        if self._use_wandb:
            wandb.log({"commitment_score": score, "step": step})

    def flush(self) -> None:
        """Write all accumulated rows to W&B as a Table artifact."""
        if not self._use_wandb:
            return
        table = wandb.Table(columns=self.COLUMNS, data=self._rows)
        try:
            wandb.log({"trajectory": table})
        except FileNotFoundError as e:
            # W&B Windows bug: temp media directory not created before write.
            # Per-step metrics and summary are already logged; table is best-effort.
            # Full trajectory is preserved in the local JSONL file.
            print(f"  [wandb] trajectory table skipped (temp dir issue): {e}")
            if self._local_path:
                print(f"  [wandb] full trajectory saved to {self._local_path}")

        if self._commitment_scores:
            scores = list(self._commitment_scores.values())
            wandb.summary["commitment_score_initial"] = scores[0]
            wandb.summary["commitment_score_final"] = scores[-1]
            wandb.summary["commitment_score_mean"] = sum(scores) / len(scores)
            wandb.summary["commitment_maintained"] = scores[-1] >= scores[0] - 1


def init_run(
    hypothesis: Hypothesis,
    llm_config: LLMConfig,
    exp_config: ExperimentConfig,
    wandb_api_key: Optional[str] = None,
    no_wandb: bool = False,
) -> TrajectoryLogger:
    """Initialize a W&B run and return a fresh TrajectoryLogger.

    Pass no_wandb=True to skip W&B entirely (useful for testing LLM behaviour).
    """
    if no_wandb:
        print("W&B disabled — logging to stdout only.")
        os.makedirs("artifacts/trajectories", exist_ok=True)
        local_path = f"artifacts/trajectories/{exp_config.condition}_{hypothesis.id}_local.jsonl"
        return TrajectoryLogger(use_wandb=False, local_path=local_path)

    if wandb_api_key:
        wandb.login(key=wandb_api_key, relogin=True)
    run_name = exp_config.run_name or f"{exp_config.condition}_{hypothesis.id}"
    wandb.init(
        project=exp_config.wandb_project,
        entity=exp_config.wandb_entity,
        name=run_name,
        config={
            "condition": exp_config.condition,
            "hypothesis_id": hypothesis.id,
            "hypothesis_statement": hypothesis.statement,
            "commitment_steps": exp_config.commitment_steps,
            "model": llm_config.model,
            "base_url": llm_config.base_url,
            "temperature": llm_config.temperature,
            "seed": exp_config.seed,
        },
        reinit="finish_previous",
    )
    os.makedirs("artifacts/trajectories", exist_ok=True)
    local_path = f"artifacts/trajectories/{run_name}_{wandb.run.id}.jsonl"
    return TrajectoryLogger(use_wandb=True, local_path=local_path)


def finish_run(logger: TrajectoryLogger) -> None:
    logger.flush()
    if logger._use_wandb:
        wandb.finish()
