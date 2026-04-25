"""NAS-Bench-201 topology critic for Phase 2.

Unlike the generic CriticAgent which challenges the hypothesis principle,
this critic challenges the specific CELL TOPOLOGY — which edges are active,
which are dead, and why those choices might be suboptimal compared to
alternatives that still satisfy the hypothesis constraint.

This forces the committed agent to actually explore different cells within
its constrained subspace rather than just defending the hypothesis principle
while holding the same cell.
"""

from haas.llm_client import LLMClient, NAS_CHALLENGE_SCHEMA
from haas.nas_bench import (
    parse_cell_string,
    EDGE_ENDPOINTS,
    OPERATIONS,
    HYPOTHESIS_CONSTRAINT_DESCRIPTIONS,
)


def _describe_cell(cell_str: str) -> str:
    """Produce a human-readable description of a cell's topology."""
    try:
        ops = parse_cell_string(cell_str)
    except ValueError:
        return f"(unparseable cell: {cell_str})"

    lines = []
    for i, (op, (src, dst)) in enumerate(zip(ops, EDGE_ENDPOINTS)):
        status = "ACTIVE" if op != "none" else "dead "
        lines.append(f"  Edge {i+1} (node {src}->node {dst}): {status}  [{op}]")

    active = [op for op in ops if op != "none"]
    dead = [i+1 for i, op in enumerate(ops) if op == "none"]
    lines.append(f"  Active edges: {len(active)}/6 -- {', '.join(active) if active else 'none'}")
    if dead:
        lines.append(f"  Dead edges: positions {dead}")
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are a NAS-Bench-201 architecture expert critiquing specific cell topology choices.

Your role is NOT to challenge the researcher's hypothesis (that is off-limits).
Your role IS to challenge the specific cell they proposed within their constraint:
- Question which edges are active vs dead and why
- Point out information flow bottlenecks or redundancies in the specific topology
- Suggest alternative configurations that STAY WITHIN the constraint but explore
  different trade-offs (e.g. different active positions, different operation choices)

You understand NAS-Bench-201 cell topology deeply:
- The 4-node DAG has 3 information paths from input (node 0) to output (node 3):
    Short:  0→3  (direct)
    Medium: 0→1→3  or  0→2→3
    Long:   0→1→2→3
- Active edges on shorter paths give stronger gradient flow to the output
- Dead edges create sparse paths but may create bottlenecks
- The output node (node 3) needs at least one active incoming edge to function

Keep your critique to 3-4 sentences. Be specific — cite exact edge positions.
"""

_CHALLENGE_PROMPT = """\
The researcher is committed to this hypothesis:
{hypothesis_statement}

This constrains them to:
{constraint_description}

Their current cell string: {cell_str}
CIFAR-10 accuracy of this cell: {acc}

Topology breakdown:
{topology_description}

Challenge the SPECIFIC TOPOLOGY CHOICES — not the hypothesis itself.
Use the accuracy as evidence: if the score is low, point to specific edges
that may be responsible. If the score is high, challenge whether a different
topology within the constraint could push further. Suggest an alternative
configuration within the same constraint. /no_think
"""


class NASCriticAgent:
    def __init__(self, client: LLMClient):
        self._client = client

    def challenge(
        self,
        cell_str: str,
        hypothesis_id: str,
        hypothesis_statement: str,
        acc: float | None = None,
    ) -> str:
        """Generate a topology-specific challenge for the given cell."""
        constraint_desc = HYPOTHESIS_CONSTRAINT_DESCRIPTIONS.get(
            hypothesis_id, "No specific constraint defined."
        )
        topology_desc = _describe_cell(cell_str)
        acc_str = f"{acc:.2f}%" if acc is not None else "not evaluated"

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _CHALLENGE_PROMPT.format(
                    hypothesis_statement=hypothesis_statement,
                    constraint_description=constraint_desc,
                    cell_str=cell_str,
                    acc=acc_str,
                    topology_description=topology_desc,
                ),
            },
        ]
        result = self._client.chat_json(messages, NAS_CHALLENGE_SCHEMA)
        return (
            f"[Topology] {result['topology_observation']} "
            f"— {result['challenge']} "
            f"Suggested direction: {result['suggested_direction']}"
        )
