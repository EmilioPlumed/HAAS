"""NAS-Bench-201 baseline agent for Phase 2.

Standard generate-critique-refine loop, no commitment injection.
When challenged, immediately incorporates the critique — including adopting
operations that may violate the hypothesis domain.

Every response includes a valid NAS-Bench-201 cell string.
"""

from typing import Dict, List, Optional

from haas.hypotheses import Hypothesis
from haas.llm_client import (
    LLMClient,
    NAS_PROPOSAL_SCHEMA,
    NAS_DEFENSE_SCHEMA,
    NAS_FINAL_SCHEMA,
)


def _dict_to_text(d: dict) -> str:
    return "\n\n".join(f"{k}:\n{v}" for k, v in d.items())


_SYSTEM_PROMPT = """\
You are an AI research agent exploring the NAS-Bench-201 cell search space.
Your goal is to generate high-quality cell architectures and improve them
iteratively based on expert feedback.

You are working within the NAS-Bench-201 topology search space. Each architecture
is a 4-node directed acyclic graph (DAG) with 6 directed edges. You must express
every architectural proposal as a valid NAS-Bench-201 cell string.

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

When given a critique, thoughtfully incorporate it to refine your cell.
Prioritize proposals that align with established NAS best practices and
empirical evidence from the NAS-Bench-201 literature.

Example valid cell string: |nor_conv_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|none~0|nor_conv_3x3~1|avg_pool_3x3~2|
"""

_INITIAL_PROMPT = """\
Generate a NAS-Bench-201 cell architecture. The cell should explore the following
architectural direction (use this as a starting point, not a constraint):

{topic_hint}

Requirements:
- Output a valid cell string in the exact format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
- Every operation must be one of: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
- Explain each of the 6 edge choices and their rationale
/no_think
"""

_REFINE_PROMPT = """\
Your previous cell scored {acc} on CIFAR-10.

You have received the following critique of your cell proposal:

{critique}

Revise your cell to address this critique and improve the architecture.

Requirements:
- Output a valid cell string in the exact format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
- Every operation must be one of: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
- Use the accuracy feedback and the critique to reason about what to change
- Explain what changed and why
/no_think
"""

_FINALIZE_PROMPT = """\
The refinement phase is complete. You have proposed and evaluated the following cells:

{trajectory_summary}

Your task now is to SELECT THE BEST cell from the ones you explored above.

Steps:
1. Review the accuracy of each cell and what you learned from refining it.
2. Pick the cell with the highest empirical performance.
3. Explain WHY that cell beats the alternatives — cite specific accuracies and trade-offs.
4. If you believe a novel cell would outperform everything you tried, you may propose it —
   but justify specifically what makes it better than your highest-accuracy cell above.

Requirements:
- Output a valid cell string in the exact format: |op~0|+|op~0|op~1|+|op~0|op~1|op~2|
- Every operation must be one of: none, skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3
- The design_rationale must reference specific accuracy numbers from the trajectory
- State the one key empirical question this cell will answer
/no_think
"""


class NASBaselineAgent:
    def __init__(self, hypothesis: Hypothesis, client: LLMClient):
        topic_hint = f"Focus on architectures related to: {hypothesis.statement}"
        self._client = client
        self._topic_hint = topic_hint
        self._history: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT}
        ]
        self._explored_cells: List[str] = []

    def initial_proposal(self) -> dict:
        result = self._step_json(
            _INITIAL_PROMPT.format(topic_hint=self._topic_hint), NAS_PROPOSAL_SCHEMA
        )
        self._record_cell(result, label="Step 0 (initial)", acc=None)
        return result

    def refine(self, critique: str, acc: Optional[float] = None) -> dict:
        acc_str = f"{acc:.2f}%" if acc is not None else "not yet evaluated"
        result = self._step_json(
            _REFINE_PROMPT.format(critique=critique, acc=acc_str), NAS_DEFENSE_SCHEMA
        )
        step = len(self._explored_cells)
        self._record_cell(result, label=f"Step {step}", acc=None)
        return result

    def finalize(self) -> dict:
        summary = self._build_trajectory_summary()
        return self._step_json(
            _FINALIZE_PROMPT.format(trajectory_summary=summary), NAS_FINAL_SCHEMA
        )

    def record_acc(self, acc: Optional[float]) -> None:
        """Patch the accuracy into the most recently recorded cell, once evaluated."""
        if not self._explored_cells:
            return
        entry = self._explored_cells[-1]
        acc_str = f"{acc:.2f}%" if acc is not None else "not evaluated"
        if "| acc:" not in entry:
            first_line, *rest = entry.split("\n")
            self._explored_cells[-1] = f"{first_line} | acc: {acc_str}\n" + "\n".join(rest)

    def _record_cell(self, result: dict, label: str, acc: Optional[float] = None) -> None:
        cell = result.get("cell_string", "INVALID")
        rationale = result.get("op_rationale") or result.get("changes_made") or result.get("refinement_summary") or ""
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
