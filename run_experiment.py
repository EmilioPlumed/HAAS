#!/usr/bin/env python3
"""CLI entry point for HAAS experiments.

Examples
--------
# HAAS condition, hypothesis "no_skip", local Ollama
python run_experiment.py --condition haas --hypothesis no_skip

# Baseline condition, same hypothesis
python run_experiment.py --condition baseline --hypothesis no_skip

# Override model and endpoint
python run_experiment.py --condition haas --hypothesis no_attention \\
    --model mistral-small:latest --base-url http://localhost:11434/v1

# Use OpenAI API
python run_experiment.py --condition haas --hypothesis pure_conv \\
    --base-url https://api.openai.com/v1 --api-key $OPENAI_API_KEY \\
    --model gpt-4o

# Run both conditions for one hypothesis in sequence
python run_experiment.py --condition haas     --hypothesis shared_weights
python run_experiment.py --condition baseline --hypothesis shared_weights

# List available hypotheses
python run_experiment.py --list-hypotheses
"""

import argparse
import sys

from haas.config import ExperimentConfig, LLMConfig
from haas.hypotheses import HYPOTHESIS_POOL
from haas.loop import run_experiment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a HAAS or baseline architecture search experiment."
    )
    p.add_argument(
        "--condition",
        choices=["haas", "baseline"],
        default="haas",
        help="Agent condition: committed (haas) or standard (baseline). Default: haas",
    )
    p.add_argument(
        "--hypothesis",
        dest="hypothesis_id",
        choices=list(HYPOTHESIS_POOL.keys()),
        default=None,
        help="Hypothesis ID from the pool. Omit to pick at random.",
    )
    p.add_argument(
        "--commitment-steps",
        type=int,
        default=5,
        help="Number of develop-and-defend steps before evaluation phase. Default: 5",
    )
    p.add_argument(
        "--model",
        default="qwen3.5:9b-64k",
        help="Model name for the OpenAI-compatible endpoint. Default: qwen3.5:9b-64k",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="LLM API base URL. Default: http://localhost:11434/v1 (Ollama)",
    )
    p.add_argument(
        "--api-key",
        default="ollama",
        help="API key for the endpoint. Default: 'ollama' (no auth needed for Ollama)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Default: 0.7",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens for JSON content output. Default: None (no limit — required for Ollama grammar-constrained generation).",
    )
    p.add_argument(
        "--wandb-project",
        default="haas",
        help="Weights & Biases project name. Default: haas",
    )
    p.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity (username or team). Optional.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Human-readable W&B run name. Auto-generated if omitted.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for hypothesis selection. Default: 42",
    )
    p.add_argument(
        "--wandb-api-key",
        default=None,
        help="Weights & Biases API key. Pass directly to avoid interactive login prompt.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging entirely. Useful for testing LLM behaviour.",
    )
    p.add_argument(
        "--list-hypotheses",
        action="store_true",
        help="Print available hypotheses and exit.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_hypotheses:
        print("\nAvailable hypotheses:\n")
        for h in HYPOTHESIS_POOL.values():
            print(f"  {h.id}")
            print(f"    {h.statement}")
            print(f"    [rationale: {h.rationale}]")
            print()
        sys.exit(0)

    llm_config = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    exp_config = ExperimentConfig(
        commitment_steps=args.commitment_steps,
        condition=args.condition,
        hypothesis_id=args.hypothesis_id,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        run_name=args.run_name,
    )

    summary = run_experiment(llm_config, exp_config, wandb_api_key=args.wandb_api_key, no_wandb=args.no_wandb)

    print("\n" + "=" * 60)
    print("Run complete.")
    print(f"  condition              : {summary['condition']}")
    print(f"  hypothesis             : {summary['hypothesis_id']}")
    print(f"  final commitment score : {summary['final_commitment_score']}/10")
    print("=" * 60)


if __name__ == "__main__":
    main()
