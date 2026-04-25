"""NAS-Bench-201 committed agent for Phase 2.

Behavioral contract (same as committed_agent.py):
  - Every response includes a valid NAS-Bench-201 cell string.
  - The cell string must satisfy the hypothesis constraint.
  - When challenged, steelmans first, then may refine implementation details.
  - Core hypothesis (and its constraint on which operations are allowed) is NON-NEGOTIABLE.

Cell format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
  - 4-node DAG, 6 directed edges
  - ops: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
"""

from typing import Dict, List, Optional

from haas.hypotheses import Hypothesis
from haas.nas_bench import HYPOTHESIS_CONSTRAINT_DESCRIPTIONS
from haas.llm_client import (
    LLMClient,
    NAS_PROPOSAL_SCHEMA,
    NAS_DEFENSE_SCHEMA,
    NAS_FINAL_SCHEMA,
)


def _dict_to_text(d: dict) -> str:
    return "\n\n".join(f"{k}:\n{v}" for k, v in d.items())


_SYSTEM_TEMPLATE = """\
You are an AI research agent committed to the following architectural hypothesis
for the NAS-Bench-201 cell search space.

========== COMMITTED HYPOTHESIS (NON-NEGOTIABLE) ==========
{statement}
============================================================

You are working within the NAS-Bench-201 topology search space. Each architecture
is a 4-node directed acyclic graph (DAG) with 6 edges. You must express every
architectural proposal as a valid NAS-Bench-201 cell string.

CELL STRING FORMAT: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|

The number after ~ is the SOURCE NODE INDEX of that edge. It is fixed:
  Group 1 (→ node 1): |op~0|                     — 1 edge, always ~0
  Group 2 (→ node 2): |op~0|op~1|                — 2 edges, always ~0 then ~1
  Group 3 (→ node 3): |op~0|op~1|op~2|           — 3 edges, always ~0 then ~1 then ~2

CRITICAL: the ~N values are NOT a choice — they are fixed positional labels.
  ~0 means "from node 0", ~1 means "from node 1", ~2 means "from node 2".
  You must use exactly: ~0 | ~0 ~1 | ~0 ~1 ~2 — in that order, always.

The 6 positions (left to right) encode these directed edges:
  Position 1 (~0): edge from node 0 → node 1
  Position 2 (~0): edge from node 0 → node 2
  Position 3 (~1): edge from node 1 → node 2
  Position 4 (~0): edge from node 0 → node 3
  Position 5 (~1): edge from node 1 → node 3
  Position 6 (~2): edge from node 2 → node 3

AVAILABLE OPERATIONS (you must use exactly these names):
  none            — dead edge, no information flow
  skip_connect    — identity shortcut (skip connection)
  nor_conv_1x1    — 1×1 pointwise convolution
  nor_conv_3x3    — 3×3 convolution
  avg_pool_3x3    — 3×3 average pooling

CONSTRAINT FROM YOUR HYPOTHESIS:
{constraint_description}

Rules you must follow:
1. Every cell string you output MUST satisfy the hypothesis constraint above.
2. When challenged on your cell topology, defend your hypothesis principle FIRST —
   then use the critique as a reason to explore a DIFFERENT cell configuration.
3. You MUST propose a DIFFERENT cell string each round — not the same cell repeated.
   Your commitment is to the CONSTRAINT (which operations are forbidden), not to
   any specific cell. Actively search the constrained subspace.
4. Think about NAS-Bench-201 topology when choosing a new cell:
   - Which information paths (0→3, 0→1→3, 0→2→3, 0→1→2→3) should be active?
   - Where should the active operations be placed for best feature reuse?
   - Are there dead edges creating bottlenecks you should address?
5. You are a researcher committed to this hypothesis AND to finding the best
   architecture within it. Treat every challenge as an opportunity to explore
   a different corner of your constrained subspace.

Example valid cell string: |nor_conv_3x3~0|+|nor_conv_1x1~0|none~1|+|none~0|nor_conv_3x3~1|nor_conv_1x1~2|
"""

_INITIAL_PROMPT = """\
Generate your first NAS-Bench-201 cell architecture that develops and instantiates
your committed hypothesis.

Requirements:
- Output a valid cell string in the exact format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
- Every operation must be one of: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
- The cell MUST satisfy your hypothesis constraint (check each edge operation)
- Explain each of the 6 edge choices and how they follow from the hypothesis

Before writing the cell_string field, read back each of the 6 operations you have
chosen and confirm that every one satisfies your constraint. If any violates it,
correct it before outputting.
/no_think
"""

_STEELMAN_PROMPT = """\
Your previous cell scored {acc} on CIFAR-10.

You have received the following challenge about your cell topology:

{challenge}

Step 1 — Defend your hypothesis principle: explain why the constraint itself
is valid, regardless of the specific cell.

Step 2 — Explore: propose a DIFFERENT cell string that improves on your
previous result. Specifically:
- Use the accuracy feedback and the critique to reason about what to change
- Change at least 1-2 edge choices from your previous cell
- Stay within your hypothesis constraint (no forbidden operations)
- Reason about the NAS-Bench-201 topology: which paths carry information,
  where are the bottlenecks, what does the output node receive?

Your previous cell is a starting point, not a commitment. The hypothesis
constraint is the commitment. Explore the constrained subspace.

Before writing the cell_string field, read back each of the 6 operations you have
chosen and confirm that every one satisfies your constraint. If any violates it,
correct it before outputting.
/no_think
"""

_FINALIZE_PROMPT = """\
The exploration phase is complete. You have proposed and evaluated the following cells:

{trajectory_summary}

Your task now is to SELECT THE BEST cell from the ones you explored above.

Steps:
1. Review the accuracy of each cell and what you learned from defending it.
2. Pick the cell with the best balance of empirical performance and hypothesis alignment.
3. Explain WHY that cell beats the alternatives — cite specific accuracies, edges, and trade-offs.
4. If you believe a novel cell would outperform everything you tried, you may propose it —
   but justify specifically what makes it better than your highest-accuracy cell above.

Requirements:
- Output a valid cell string in the exact format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
- The cell MUST satisfy the hypothesis constraint
- The design_rationale must reference specific accuracy numbers from the trajectory
- State the one key empirical question this final cell will resolve via NAS-Bench-201
/no_think
"""


class NASCommittedAgent:
    def __init__(self, hypothesis: Hypothesis, client: LLMClient):
        self._hypothesis = hypothesis
        self._client = client
        constraint_desc = HYPOTHESIS_CONSTRAINT_DESCRIPTIONS.get(
            hypothesis.id, "No specific constraint defined."
        )
        self._history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": _SYSTEM_TEMPLATE.format(
                    statement=hypothesis.statement,
                    constraint_description=constraint_desc,
                ),
            }
        ]
        self._explored_cells: List[str] = []

    def initial_proposal(self) -> dict:
        result = self._step_json(_INITIAL_PROMPT, NAS_PROPOSAL_SCHEMA)
        self._record_cell(result, label="Step 0 (initial)", acc=None)
        return result

    def steelman_and_refine(self, challenge: str, acc: Optional[float] = None) -> dict:
        """Challenge the current cell (whose accuracy is `acc`) and propose a new one."""
        acc_str = f"{acc:.2f}%" if acc is not None else "not yet evaluated"
        result = self._step_json(
            _STEELMAN_PROMPT.format(challenge=challenge, acc=acc_str), NAS_DEFENSE_SCHEMA
        )
        step = len(self._explored_cells)
        self._record_cell(result, label=f"Step {step}", acc=None)
        return result

    def record_acc(self, acc: Optional[float]) -> None:
        """Patch the accuracy into the most recently recorded cell, once evaluated."""
        if not self._explored_cells:
            return
        entry = self._explored_cells[-1]
        acc_str = f"{acc:.2f}%" if acc is not None else "not evaluated"
        if "| acc:" not in entry:
            first_line, *rest = entry.split("\n")
            self._explored_cells[-1] = f"{first_line} | acc: {acc_str}\n" + "\n".join(rest)

    def finalize(self) -> dict:
        summary = self._build_trajectory_summary()
        return self._step_json(_FINALIZE_PROMPT.format(trajectory_summary=summary), NAS_FINAL_SCHEMA)

    def _record_cell(self, result: dict, label: str, acc: Optional[float] = None) -> None:
        cell = result.get("cell_string", "INVALID")
        rationale = result.get("op_rationale") or result.get("refinement_summary") or ""
        acc_str = f" | acc: {acc:.2f}%" if acc is not None else ""
        self._explored_cells.append(f"{label}: {cell}{acc_str}\n  Rationale: {rationale[:200]}")

    def _build_trajectory_summary(self) -> str:
        if not self._explored_cells:
            return "(no cells recorded)"
        return "\n\n".join(self._explored_cells)

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def _step_json(self, user_message: str, schema: dict) -> dict:
        self._history.append({"role": "user", "content": user_message})
        result = self._client.chat_json(self._history, schema)
        self._history.append({"role": "assistant", "content": _dict_to_text(result)})
        return result
