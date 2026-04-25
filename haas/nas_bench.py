"""NAS-Bench-201 / NATS-Bench Topology Search Space interface.

The NAS-Bench-201 search space (NATS-Bench TSS) encodes 4-node DAGs with
5 possible operations on each of the 6 directed edges:

  Operations:
    none            — dead edge (no information flow)
    skip_connect    — identity shortcut
    nor_conv_1x1    — 1×1 pointwise convolution
    nor_conv_3x3    — 3×3 convolution
    avg_pool_3x3    — 3×3 average pooling

  Cell topology (4 nodes, 6 edges):
    node 0 → node 1  (edge 0)
    node 0 → node 2  (edge 1)
    node 1 → node 2  (edge 2)
    node 0 → node 3  (edge 3)
    node 1 → node 3  (edge 4)
    node 2 → node 3  (edge 5)

  Cell string format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
    - Group 1 (node 1):  edge from node 0
    - Group 2 (node 2):  edges from nodes 0 and 1
    - Group 3 (node 3):  edges from nodes 0, 1, and 2

  Example: |nor_conv_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|none~0|nor_conv_3x3~1|avg_pool_3x3~2|

  Total architectures: 5^6 = 15,625
"""

import itertools
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# NOTE: nats_bench emits a VisibleDeprecationWarning on first data access due to a
# numpy 2.4 incompatibility (dtype align kwarg deprecated). This warning is emitted
# from inside pickle.load in nats_bench C extension code and cannot be suppressed via
# Python's warnings filter. It fires once per process then silences. Not actionable.

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
ACTIVE_OPS = ["skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
NUM_EDGES = 6
NUM_ARCHS = 5 ** 6  # 15,625

# Edge index → (source_node, dest_node)
EDGE_ENDPOINTS = [
    (0, 1),  # edge 0
    (0, 2),  # edge 1
    (1, 2),  # edge 2
    (0, 3),  # edge 3
    (1, 3),  # edge 4
    (2, 3),  # edge 5
]

# Default data directory
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"

# ---------------------------------------------------------------------------
# Cell string encoding / decoding
# ---------------------------------------------------------------------------

def build_cell_string(ops: List[str]) -> str:
    """Build a NAS-Bench-201 cell string from a list of 6 operations.

    ops[i] is the operation on edge i, in the order:
      [e(0→1), e(0→2), e(1→2), e(0→3), e(1→3), e(2→3)]
    """
    if len(ops) != NUM_EDGES:
        raise ValueError(f"Expected {NUM_EDGES} operations, got {len(ops)}")
    for op in ops:
        if op not in OPERATIONS:
            raise ValueError(f"Unknown operation: {op!r}. Must be one of {OPERATIONS}")
    # Group by destination node
    # Node 1: edge 0 (from 0)
    g1 = f"|{ops[0]}~0|"
    # Node 2: edges 1 (from 0), 2 (from 1)
    g2 = f"|{ops[1]}~0|{ops[2]}~1|"
    # Node 3: edges 3 (from 0), 4 (from 1), 5 (from 2)
    g3 = f"|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|"
    return f"{g1}+{g2}+{g3}"


def parse_cell_string(cell_str: str) -> List[str]:
    """Parse a NAS-Bench-201 cell string into a list of 6 operations.

    Returns ops in order [e(0→1), e(0→2), e(1→2), e(0→3), e(1→3), e(2→3)].
    Raises ValueError if the string is malformed or contains unknown operations.
    """
    # Extract all op~node tokens
    tokens = re.findall(r"(\w+)~\d+", cell_str)
    if len(tokens) != NUM_EDGES:
        raise ValueError(
            f"Expected {NUM_EDGES} op tokens in cell string, got {len(tokens)}: {cell_str!r}"
        )
    for op in tokens:
        if op not in OPERATIONS:
            raise ValueError(
                f"Unknown operation {op!r} in cell string. Valid ops: {OPERATIONS}"
            )
    return tokens


def canonicalize_cell_string(cell_str: str) -> Optional[str]:
    """Parse a (possibly malformed) cell string and rebuild it in canonical form.

    LLMs frequently output the correct operation names but use the wrong source
    node index (e.g. ``none~0`` instead of ``none~1`` in group 2). This function
    extracts the 6 operation names in positional order and rebuilds the string
    with the correct ``~N`` source-node indices.

    Returns the canonical cell string, or None if the ops cannot be extracted
    (wrong number of tokens or unknown operation name).
    """
    try:
        ops = parse_cell_string(cell_str)   # extracts ops, ignores ~N values
        return build_cell_string(ops)       # rebuilds with correct ~N indices
    except ValueError:
        return None


def validate_cell_string(cell_str: str) -> bool:
    """Return True if cell_str is a well-formed NAS-Bench-201 architecture string.

    Accepts strings where the source-node indices (``~N``) may be wrong — the
    canonical form is reconstructed for the check. Use ``canonicalize_cell_string``
    to get the corrected string before passing to the API.
    """
    return canonicalize_cell_string(cell_str) is not None


# ---------------------------------------------------------------------------
# Hypothesis constraints
# ---------------------------------------------------------------------------

def _make_constraint(allowed_ops: List[str]) -> Callable[[List[str]], bool]:
    """Return a constraint function that checks all ops are in allowed_ops."""
    allowed = set(allowed_ops)
    def constraint(ops: List[str]) -> bool:
        return all(op in allowed for op in ops)
    return constraint


def _sparse_constraint(max_active: int = 2) -> Callable[[List[str]], bool]:
    """Return a constraint that requires at most max_active non-'none' edges."""
    def constraint(ops: List[str]) -> bool:
        active = sum(1 for op in ops if op != "none")
        return active <= max_active
    return constraint


# Constraint per hypothesis_id — each is a callable: List[str] → bool
HYPOTHESIS_CONSTRAINTS: Dict[str, Callable[[List[str]], bool]] = {
    # No skip_connect anywhere in the cell
    "no_skip": _make_constraint(["none", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]),
    # Only convolutions (no skip, no pooling)
    "pure_conv": _make_constraint(["none", "nor_conv_1x1", "nor_conv_3x3"]),
    # At most 2 active edges — maximally sparse cell
    "sparse_cell": _sparse_constraint(max_active=2),
    # Only 1×1 conv and none — no 3×3 anything
    "no_3x3": _make_constraint(["none", "nor_conv_1x1"]),
}

# Human-readable description of each constraint (for agent prompts)
HYPOTHESIS_CONSTRAINT_DESCRIPTIONS: Dict[str, str] = {
    "no_skip": (
        "FORBIDDEN operations: skip_connect\n"
        "ALLOWED operations: none, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3\n"
        "Every edge must use one of the 4 allowed ops. No skip_connect anywhere."
    ),
    "pure_conv": (
        "FORBIDDEN operations: skip_connect, avg_pool_3x3\n"
        "ALLOWED operations: none, nor_conv_1x1, nor_conv_3x3\n"
        "Only convolutional operations — no skip shortcuts, no pooling."
    ),
    "sparse_cell": (
        "SPARSITY REQUIREMENT: at most 2 of the 6 edges may be non-'none'.\n"
        "At least 4 edges must be 'none'. The 1–2 active edges can use any "
        "operation: skip_connect, nor_conv_1x1, nor_conv_3x3, or avg_pool_3x3.\n"
        "Example valid: 5× none + 1× nor_conv_3x3; or 4× none + 1× nor_conv_1x1 + 1× avg_pool_3x3."
    ),
    "no_3x3": (
        "FORBIDDEN operations: nor_conv_3x3, avg_pool_3x3, skip_connect\n"
        "ALLOWED operations: none, nor_conv_1x1\n"
        "Only 1×1 pointwise convolutions and dead edges — no spatial kernels of any kind."
    ),
}


def satisfies_constraint(ops: List[str], hypothesis_id: str) -> bool:
    """Return True if ops satisfies the constraint for the given hypothesis."""
    if hypothesis_id not in HYPOTHESIS_CONSTRAINTS:
        raise ValueError(f"No constraint defined for hypothesis {hypothesis_id!r}")
    return HYPOTHESIS_CONSTRAINTS[hypothesis_id](ops)


# ---------------------------------------------------------------------------
# Cell enumeration
# ---------------------------------------------------------------------------

def enumerate_constrained_cells(hypothesis_id: str) -> List[str]:
    """Return all NAS-Bench-201 cell strings satisfying the hypothesis constraint."""
    if hypothesis_id not in HYPOTHESIS_CONSTRAINTS:
        raise ValueError(f"No constraint defined for hypothesis {hypothesis_id!r}")
    constraint = HYPOTHESIS_CONSTRAINTS[hypothesis_id]
    result = []
    for ops in itertools.product(OPERATIONS, repeat=NUM_EDGES):
        ops_list = list(ops)
        if constraint(ops_list):
            result.append(build_cell_string(ops_list))
    return result


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def hamming_distance(cell_a: str, cell_b: str) -> int:
    """Count the number of edges where cell_a and cell_b differ."""
    ops_a = parse_cell_string(cell_a)
    ops_b = parse_cell_string(cell_b)
    return sum(a != b for a, b in zip(ops_a, ops_b))


def mean_pairwise_hamming(cells: List[str]) -> float:
    """Mean pairwise Hamming distance across all pairs in the list."""
    if len(cells) < 2:
        return 0.0
    total, count = 0, 0
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            try:
                total += hamming_distance(cells[i], cells[j])
                count += 1
            except ValueError:
                continue
    return total / count if count > 0 else 0.0


def unique_cell_count(cells: List[str]) -> int:
    """Count the number of distinct valid cell strings in the list."""
    valid = set()
    for c in cells:
        if validate_cell_string(c):
            valid.add(c)
    return len(valid)


def hypothesis_compliance_rate(cells: List[str], hypothesis_id: str) -> float:
    """Fraction of cells that satisfy the hypothesis constraint (0.0–1.0)."""
    if not cells:
        return 0.0
    compliant = 0
    for c in cells:
        try:
            ops = parse_cell_string(c)
            if satisfies_constraint(ops, hypothesis_id):
                compliant += 1
        except ValueError:
            continue
    return compliant / len(cells)


# ---------------------------------------------------------------------------
# NAS-Bench-201 / NATS-Bench API
# ---------------------------------------------------------------------------

def load_api(data_path: Optional[str] = None, fast_mode: bool = True):
    """Load the NATS-Bench topology search space API.

    Args:
        data_path: Path to the NATS-TSS data directory or .tar file.
                   Defaults to <project_root>/data/NATS-tss-v1_0-3ffb9-simple.tar
        fast_mode: If True, loads data lazily (recommended; faster startup).

    Returns:
        NATSBench API object, or None if the data file is missing.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=Warning)
        from nats_bench import create  # type: ignore

    if data_path is None:
        candidates = [
            _DEFAULT_DATA_DIR / "NATS-tss-v1_0-3ffb9-simple",
            _DEFAULT_DATA_DIR / "NATS-tss-v1_0-3ffb9-simple.tar",
        ]
        for candidate in candidates:
            if candidate.exists():
                data_path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"NAS-Bench-201 data not found in {_DEFAULT_DATA_DIR}.\n"
                "Run: python -m gdown 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU "
                "-O data/NATS-tss-v1_0-3ffb9-simple.tar"
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning)
        return create(data_path, "tss", fast_mode=fast_mode, verbose=False)


def query_accuracy(
    api,
    cell_str: str,
    dataset: str = "cifar10",
    hp: str = "200",
    metric: str = "test",
) -> Optional[float]:
    """Query NAS-Bench-201 accuracy for a given cell string.

    Args:
        api:      NATS-Bench API object from load_api().
        cell_str: NAS-Bench-201 cell string.
        dataset:  One of 'cifar10', 'cifar100', 'ImageNet16-120'.
        hp:       Training budget in epochs. '200' uses the full NAS-Bench-201 budget.
        metric:   'test' or 'valid'.

    Returns:
        Accuracy as a float (e.g. 93.15), or None if the architecture is not found.
    """
    try:
        idx = api.query_index_by_arch(cell_str)
        if idx == -1:
            return None
        info = api.get_more_info(idx, dataset, hp=hp, is_random=False)
        key = f"{metric}-accuracy"
        return info.get(key)
    except Exception:
        return None


def query_all_datasets(
    api,
    cell_str: str,
    hp: str = "200",
) -> Dict[str, Optional[float]]:
    """Query test accuracy on all three NAS-Bench-201 datasets."""
    datasets = ["cifar10", "cifar100", "ImageNet16-120"]
    return {ds: query_accuracy(api, cell_str, dataset=ds, hp=hp) for ds in datasets}


def find_best_in_constraint(
    api,
    hypothesis_id: str,
    dataset: str = "cifar10",
    hp: str = "200",
) -> Tuple[Optional[str], Optional[float]]:
    """Find the highest-accuracy cell among all cells satisfying the hypothesis constraint.

    Returns (best_cell_string, best_accuracy). Useful as an oracle upper bound.
    """
    cells = enumerate_constrained_cells(hypothesis_id)
    best_cell, best_acc = None, -1.0
    for cell in cells:
        acc = query_accuracy(api, cell, dataset=dataset, hp=hp)
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_cell = cell
    return best_cell, best_acc if best_acc >= 0 else None
