#!/usr/bin/env python3
"""CLI entry point for Phase 2 HAAS experiments (NAS-Bench-201 search trajectory).

Phase 2 tests whether committed agents explore structurally different regions of
the NAS-Bench-201 search space than baseline agents, and whether those regions
contain higher-performing architectures.

Examples
--------
# HAAS condition, no_3x3 hypothesis
python run_phase2.py --condition haas --hypothesis no_3x3

# Both conditions for sparse_cell (run sequentially)
python run_phase2.py --condition haas     --hypothesis sparse_cell
python run_phase2.py --condition baseline --hypothesis sparse_cell

# All Phase 2 hypotheses, HAAS condition
python run_phase2.py --condition haas --hypothesis no_skip
python run_phase2.py --condition haas --hypothesis pure_conv
python run_phase2.py --condition haas --hypothesis sparse_cell
python run_phase2.py --condition haas --hypothesis no_3x3

# Run without benchmark evaluation (cell strings proposed but not scored)
python run_phase2.py --condition haas --hypothesis no_3x3 --no-benchmark

# Run without W&B logging
python run_phase2.py --condition haas --hypothesis no_3x3 --no-wandb

# List Phase 2 hypotheses
python run_phase2.py --list-hypotheses
"""

import argparse
import sys

from haas.config import ExperimentConfig, LLMConfig
from haas.hypotheses import HYPOTHESIS_POOL
from haas.phase2_loop import run_phase2_experiment, PHASE2_HYPOTHESES
from haas.nas_bench import (
    enumerate_constrained_cells,
    find_best_in_constraint,
    load_api,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a Phase 2 HAAS or baseline NAS-Bench-201 experiment."
    )
    p.add_argument(
        "--condition",
        choices=["haas", "baseline"],
        default="haas",
        help="Agent condition. Default: haas",
    )
    p.add_argument(
        "--hypothesis",
        dest="hypothesis_id",
        choices=PHASE2_HYPOTHESES,
        default=None,
        help="Hypothesis ID. Must be one of the Phase 2 NAS-Bench-201 hypotheses.",
    )
    p.add_argument(
        "--commitment-steps",
        type=int,
        default=5,
        help="Number of develop-and-defend steps. Default: 5",
    )
    p.add_argument(
        "--model",
        default="qwen3.5:9b-64k",
        help="Model name. Default: qwen3.5:9b-64k",
    )
    p.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="LLM API base URL. Default: http://localhost:11434/v1 (Ollama)",
    )
    p.add_argument(
        "--api-key",
        default="ollama",
        help="API key. Default: 'ollama'",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Default: 0.7",
    )
    p.add_argument(
        "--wandb-project",
        default="haas-phase2",
        help="W&B project name. Default: haas-phase2",
    )
    p.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B entity. Optional.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="W&B run name. Auto-generated if omitted.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--wandb-api-key",
        default=None,
        help="W&B API key.",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging.",
    )
    p.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Skip NAS-Bench-201 evaluation (propose cell strings only).",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Path to NATS-TSS data dir or .tar file. Auto-detected if omitted.",
    )
    p.add_argument(
        "--list-hypotheses",
        action="store_true",
        help="Print Phase 2 hypotheses with constraint descriptions and exit.",
    )
    p.add_argument(
        "--oracle",
        action="store_true",
        help="Before running, print the oracle best cell for the hypothesis (requires benchmark data).",
    )
    p.add_argument(
        "--count-cells",
        action="store_true",
        help="Print the number of cells satisfying the hypothesis constraint and exit.",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_hypotheses:
        from haas.nas_bench import HYPOTHESIS_CONSTRAINT_DESCRIPTIONS
        print("\nPhase 2 NAS-Bench-201 hypotheses:\n")
        for hid in PHASE2_HYPOTHESES:
            h = HYPOTHESIS_POOL[hid]
            constraint = HYPOTHESIS_CONSTRAINT_DESCRIPTIONS.get(hid, "No constraint defined.")
            n_cells = len(enumerate_constrained_cells(hid))
            print(f"  {h.id}  ({n_cells} valid cells)")
            print(f"    Hypothesis: {h.statement[:120]}...")
            print(f"    Constraint:\n      {constraint.replace(chr(10), chr(10) + '      ')}")
            print()
        sys.exit(0)

    if args.count_cells:
        if args.hypothesis_id is None:
            for hid in PHASE2_HYPOTHESES:
                cells = enumerate_constrained_cells(hid)
                print(f"  {hid}: {len(cells)} valid cells")
        else:
            cells = enumerate_constrained_cells(args.hypothesis_id)
            print(f"  {args.hypothesis_id}: {len(cells)} valid cells")
        sys.exit(0)

    if args.hypothesis_id is None:
        parser.error("--hypothesis is required for Phase 2 experiments.")

    # Load benchmark API
    api = None
    if not args.no_benchmark:
        try:
            api = load_api(args.data_path)
            print("NAS-Bench-201 API loaded.")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Proceeding without benchmark evaluation.")

    if args.oracle and api is not None:
        print(f"\nOracle search for hypothesis '{args.hypothesis_id}'...")
        best_cell, best_acc = find_best_in_constraint(api, args.hypothesis_id)
        print(f"  Best cell: {best_cell}")
        print(f"  Best accuracy (CIFAR-10): {best_acc:.2f}%\n")

    llm_config = LLMConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
    )
    run_name = args.run_name or f"phase2_{args.condition}_{args.hypothesis_id}"
    exp_config = ExperimentConfig(
        commitment_steps=args.commitment_steps,
        condition=args.condition,
        hypothesis_id=args.hypothesis_id,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
        run_name=run_name,
    )

    summary = run_phase2_experiment(
        llm_config,
        exp_config,
        api=api,
        wandb_api_key=args.wandb_api_key,
        no_wandb=args.no_wandb,
    )

    print("\n" + "=" * 60)
    print("Phase 2 run complete.")
    print(f"  condition          : {summary['condition']}")
    print(f"  hypothesis         : {summary['hypothesis_id']}")
    print(f"  final cell         : {summary['final_cell']}")
    print(f"  final accuracy     : {summary['final_accuracy']}")
    print(f"  hypothesis compliant: {summary['final_compliant']}")
    print(f"  commitment score   : {summary['final_commitment_score']}/10")
    print(f"  trajectory diversity: {summary['trajectory_diversity']:.2f} (mean Hamming)")
    print(f"  unique cells       : {summary['unique_cells']}")
    print(f"  compliance rate    : {summary['compliance_rate']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
