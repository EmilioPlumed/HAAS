"""Baseline agent: standard generate-critique-refine loop.

No commitment injection. When challenged, immediately incorporates the critique.
All responses are JSON-schema-constrained for clean parsing.
"""

from typing import Dict, List
from haas.hypotheses import Hypothesis


def _dict_to_text(d: dict) -> str:
    """Render a result dict as plain prose for history storage.

    Storing raw JSON in the assistant turn causes the model to treat it as a
    template and echo it inside subsequent field values. Plain text avoids this.
    """
    return "\n\n".join(f"{k}:\n{v}" for k, v in d.items())
from haas.llm_client import (
    LLMClient,
    PROPOSAL_SCHEMA,
    REFINEMENT_SCHEMA,
    FINAL_PROPOSAL_SCHEMA,
)


_SYSTEM_PROMPT = """\
You are an AI research agent exploring neural architecture design. Your goal is to
generate high-quality architectural proposals for neural networks and improve them
iteratively based on expert feedback.

When given a critique, thoughtfully incorporate it to refine and improve your proposal.
Prioritize proposals that align with established best practices and empirical evidence
from the literature.

Be specific about architecture choices: layer types, depths, widths, connections,
and the rationale for each choice.
"""

_INITIAL_PROMPT = """\
Generate an architectural proposal for a neural network. Be specific: describe layer
types, connections, and dimensions. Explain the rationale for each design choice based
on established ML principles. /no_think
"""

_REFINE_PROMPT = """\
You have received the following critique of your proposal:

{critique}

Revise your proposal to address this critique and improve the architecture. /no_think
"""

_FINALIZE_PROMPT = """\
Write your final consolidated architectural proposal based on your iterative
refinement process.

Include:
1. Full architecture specification
2. Key design choices and their rationale
3. Predicted advantages
4. Remaining open questions for empirical validation
/no_think
"""


class BaselineAgent:
    def __init__(self, hypothesis: Hypothesis, client: LLMClient):
        # Topic hint ensures a fair comparison — same domain, no commitment.
        topic_hint = f"Focus your exploration on architectures related to: {hypothesis.statement}"
        self._client = client
        self._history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT + "\n\n" + topic_hint,
            }
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def initial_proposal(self) -> dict:
        return self._step_json(_INITIAL_PROMPT, PROPOSAL_SCHEMA)

    def refine(self, critique: str) -> dict:
        return self._step_json(_REFINE_PROMPT.format(critique=critique), REFINEMENT_SCHEMA)

    def finalize(self) -> dict:
        return self._step_json(_FINALIZE_PROMPT, FINAL_PROPOSAL_SCHEMA)

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step_json(self, user_message: str, schema: dict) -> dict:
        self._history.append({"role": "user", "content": user_message})
        result = self._client.chat_json(self._history, schema)
        self._history.append({"role": "assistant", "content": _dict_to_text(result)})
        return result
