"""Phase 2 trajectory analysis utilities.

Computes post-hoc metrics from completed Phase 2 runs:
  - Search space coverage: which regions did each condition explore?
  - Structural diversity: mean pairwise Hamming distance
  - Hypothesis compliance: fraction of proposals within the constrained subspace
  - Performance: benchmark accuracy of proposed cells
  - Comparison: HAAS vs baseline on each metric

Can be run standalone or imported for custom analysis.
"""

import json
from typing import Dict, List, Optional, Tuple

from haas.nas_bench import (
    OPERATIONS,
    HYPOTHESIS_CONSTRAINTS,
    validate_cell_string,
    parse_cell_string,
    build_cell_string,
    satisfies_constraint,
    hamming_distance,
    mean_pairwise_hamming,
    unique_cell_count,
    hypothesis_compliance_rate,
    enumerate_constrained_cells,
    query_accuracy,
)


# ---------------------------------------------------------------------------
# Cell extraction from W&B trajectory rows
# ---------------------------------------------------------------------------

def extract_cells_from_trajectory(rows: list, agent: str = "committed") -> List[Optional[str]]:
    """Extract cell strings from W&B trajectory table rows.

    Each row is a list: [step, phase, agent, event_type, content].
    The content field contains JSON with a [NAS]{...} suffix added by phase2_loop.

    Args:
        rows:  List of trajectory table rows.
        agent: Filter rows by agent name ('committed', 'baseline').

    Returns:
        List of cell strings in step order (None for invalid/missing).
    """
    cells = []
    for row in rows:
        if len(row) < 5:
            continue
        row_agent = row[2]
        event_type = row[3]
        content = row[4]
        if row_agent != agent:
            continue
        if event_type == "challenge":
            continue
        # Try to extract cell string from [NAS] suffix
        cell = _extract_cell_from_content(content)
        cells.append(cell)
    return cells


def _extract_cell_from_content(content: str) -> Optional[str]:
    """Pull cell_string from a content field with optional [NAS]{...} suffix."""
    # Try [NAS] suffix first
    if "[NAS]" in content:
        nas_part = content.split("[NAS]", 1)[1]
        try:
            meta = json.loads(nas_part)
            cell = meta.get("cell_string")
            if cell and validate_cell_string(cell):
                return cell
        except (json.JSONDecodeError, AttributeError):
            pass
    # Try parsing the main JSON body
    try:
        main_json = content.split("[NAS]")[0]
        data = json.loads(main_json)
        cell = data.get("cell_string")
        if cell and validate_cell_string(cell):
            return cell
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def run_metrics(
    cells: List[Optional[str]],
    hypothesis_id: Optional[str] = None,
    api=None,
    dataset: str = "cifar10",
) -> Dict:
    """Compute trajectory metrics for a single run.

    Args:
        cells:         List of cell strings proposed during the run (None = invalid/missing).
        hypothesis_id: If provided, compute compliance rate against this hypothesis.
        api:           NAS-Bench-201 API for accuracy lookup. None = skip.
        dataset:       Dataset for accuracy lookup.

    Returns dict with keys:
        valid_count, unique_count, mean_hamming, compliance_rate,
        accuracies (list), best_acc, final_cell, final_acc, final_compliant
    """
    valid = [c for c in cells if c is not None]
    unique = unique_cell_count(valid)
    diversity = mean_pairwise_hamming(valid)

    compliance = None
    if hypothesis_id and hypothesis_id in HYPOTHESIS_CONSTRAINTS:
        compliance = hypothesis_compliance_rate(valid, hypothesis_id)

    accs = []
    if api:
        for c in valid:
            acc = query_accuracy(api, c, dataset=dataset)
            accs.append(acc)
    else:
        accs = [None] * len(valid)

    best_acc = max((a for a in accs if a is not None), default=None)
    final_cell = valid[-1] if valid else None
    final_acc = accs[-1] if accs else None

    final_compliant = None
    if final_cell and hypothesis_id and hypothesis_id in HYPOTHESIS_CONSTRAINTS:
        try:
            ops = parse_cell_string(final_cell)
            final_compliant = satisfies_constraint(ops, hypothesis_id)
        except ValueError:
            final_compliant = False

    return {
        "valid_count": len(valid),
        "unique_count": unique,
        "mean_hamming": diversity,
        "compliance_rate": compliance,
        "accuracies": accs,
        "best_acc": best_acc,
        "final_cell": final_cell,
        "final_acc": final_acc,
        "final_compliant": final_compliant,
    }


# ---------------------------------------------------------------------------
# Cross-condition comparison
# ---------------------------------------------------------------------------

def compare_conditions(
    haas_cells: List[Optional[str]],
    baseline_cells: List[Optional[str]],
    hypothesis_id: str,
    api=None,
    dataset: str = "cifar10",
) -> Dict:
    """Compare HAAS and baseline trajectories on all Phase 2 metrics.

    Returns a dict with 'haas', 'baseline', and 'comparison' sub-dicts.
    """
    haas_m = run_metrics(haas_cells, hypothesis_id, api, dataset)
    base_m = run_metrics(baseline_cells, hypothesis_id, api, dataset)

    # HAAS vs baseline inter-trajectory diversity
    haas_valid = [c for c in haas_cells if c is not None]
    base_valid = [c for c in baseline_cells if c is not None]
    cross_diversity = 0.0
    cross_count = 0
    for hc in haas_valid:
        for bc in base_valid:
            try:
                cross_diversity += hamming_distance(hc, bc)
                cross_count += 1
            except ValueError:
                continue
    mean_cross = cross_diversity / cross_count if cross_count > 0 else 0.0

    # Subspace overlap: cells in both trajectories
    haas_set = set(haas_valid)
    base_set = set(base_valid)
    overlap = len(haas_set & base_set)

    # Oracle: best possible accuracy in constraint
    oracle_cells = enumerate_constrained_cells(hypothesis_id) if hypothesis_id in HYPOTHESIS_CONSTRAINTS else []
    oracle_acc = None
    oracle_cell = None
    if api and oracle_cells:
        best_acc = -1.0
        for cell in oracle_cells:
            acc = query_accuracy(api, cell, dataset=dataset)
            if acc is not None and acc > best_acc:
                best_acc = acc
                oracle_cell = cell
        if best_acc >= 0:
            oracle_acc = best_acc

    comparison = {
        "haas_vs_baseline_cross_diversity": mean_cross,
        "trajectory_overlap_cells": overlap,
        "haas_unique_not_in_baseline": len(haas_set - base_set),
        "baseline_unique_not_in_haas": len(base_set - haas_set),
        "oracle_best_cell": oracle_cell,
        "oracle_best_acc": oracle_acc,
        "hypothesis_constraint_space_size": len(oracle_cells),
    }

    return {"haas": haas_m, "baseline": base_m, "comparison": comparison}


# ---------------------------------------------------------------------------
# Operation frequency analysis
# ---------------------------------------------------------------------------

def op_frequency(cells: List[Optional[str]]) -> Dict[str, List[int]]:
    """Count how often each operation appears at each edge position across cells.

    Returns dict: op_name → list of 6 counts (one per edge position).
    """
    counts = {op: [0] * 6 for op in OPERATIONS}
    for cell in cells:
        if cell is None:
            continue
        try:
            ops = parse_cell_string(cell)
            for i, op in enumerate(ops):
                if op in counts:
                    counts[op][i] += 1
        except ValueError:
            continue
    return counts


def print_comparison_report(result: Dict, hypothesis_id: str) -> None:
    """Print a formatted comparison report to stdout."""
    h = result["haas"]
    b = result["baseline"]
    c = result["comparison"]

    print(f"\n{'='*70}")
    print(f"Phase 2 Comparison Report — hypothesis: {hypothesis_id}")
    print(f"{'='*70}\n")

    print(f"{'Metric':<35} {'HAAS':>12} {'Baseline':>12}")
    print(f"{'-'*60}")

    def row(label, hv, bv, fmt=".2f"):
        hv_str = f"{hv:{fmt}}" if hv is not None else "N/A"
        bv_str = f"{bv:{fmt}}" if bv is not None else "N/A"
        print(f"  {label:<33} {hv_str:>12} {bv_str:>12}")

    row("Valid proposals", h["valid_count"], b["valid_count"], "d")
    row("Unique cells", h["unique_count"], b["unique_count"], "d")
    row("Mean pairwise Hamming", h["mean_hamming"], b["mean_hamming"])
    row("Compliance rate", h["compliance_rate"], b["compliance_rate"])
    row("Best accuracy (CIFAR-10)", h["best_acc"], b["best_acc"])
    row("Final accuracy (CIFAR-10)", h["final_acc"], b["final_acc"])
    row("Final hypothesis compliant", int(h["final_compliant"]) if h["final_compliant"] is not None else None,
        int(b["final_compliant"]) if b["final_compliant"] is not None else None, "d")

    print(f"\n  Cross-condition diversity (mean Hamming): {c['haas_vs_baseline_cross_diversity']:.2f}")
    print(f"  Shared cells between trajectories:       {c['trajectory_overlap_cells']}")
    print(f"  Cells unique to HAAS:                    {c['haas_unique_not_in_baseline']}")
    print(f"  Cells unique to baseline:                {c['baseline_unique_not_in_haas']}")
    print(f"  Hypothesis-constrained space size:       {c['hypothesis_constraint_space_size']}")
    if c["oracle_best_acc"] is not None:
        print(f"  Oracle best accuracy:                    {c['oracle_best_acc']:.2f}%")
        print(f"  Oracle best cell:                        {c['oracle_best_cell']}")

    print(f"\n  HAAS final cell:     {h['final_cell']}")
    print(f"  Baseline final cell: {b['final_cell']}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Phase 2 trajectory data from W&B runs."
    )
    parser.add_argument("--hypothesis", required=True, choices=list(HYPOTHESIS_CONSTRAINTS.keys()))
    parser.add_argument("--haas-run", required=True, help="W&B run path for HAAS condition")
    parser.add_argument("--baseline-run", required=True, help="W&B run path for baseline condition")
    parser.add_argument("--wandb-project", default="haas-phase2")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--dataset", default="cifar10")
    args = parser.parse_args()

    import wandb
    wandb_api = wandb.Api()

    def get_cells_from_run(run_path: str, agent: str) -> List[Optional[str]]:
        run = wandb_api.run(run_path)
        for artifact in run.logged_artifacts():
            pass  # noqa — placeholder for table extraction
        # Fallback: try to get trajectory table from history
        try:
            history = run.scan_history()
            rows = []
            for h in history:
                traj = h.get("trajectory")
                if traj:
                    rows.extend(traj.get("data", []))
            return extract_cells_from_trajectory(rows, agent)
        except Exception as e:
            print(f"Warning: Could not extract cells from {run_path}: {e}")
            return []

    haas_cells = get_cells_from_run(args.haas_run, "committed")
    baseline_cells = get_cells_from_run(args.baseline_run, "baseline")

    api = None
    if not args.no_benchmark:
        try:
            from haas.nas_bench import load_api as _load_api
            api = _load_api()
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    result = compare_conditions(haas_cells, baseline_cells, args.hypothesis, api, args.dataset)
    print_comparison_report(result, args.hypothesis)
