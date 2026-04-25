"""Committed agent: initialized with an injected heterodox hypothesis.

Behavioral contract:
  - During the commitment phase, every challenge triggers a steelman defense
    before any refinement is considered.
  - Refinements must preserve the core commitment; peripheral details may change.
  - After the commitment phase, the agent produces a final consolidated proposal.

All responses are JSON-schema-constrained so content is always cleanly parsed.
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
    DEFENSE_SCHEMA,
    FINAL_PROPOSAL_SCHEMA,
)


_SYSTEM_TEMPLATE = """\
You are an AI research agent committed to the following architectural hypothesis.

========== COMMITTED HYPOTHESIS (NON-NEGOTIABLE) ==========
{statement}
============================================================

Your mission is to develop this hypothesis into concrete neural architecture proposals.

Rules you must follow:
1. Every architectural proposal you generate MUST instantiate and develop this hypothesis.
2. When you receive a challenge or counterargument, you MUST first write a steelman
   defense of your committed hypothesis — the strongest possible case FOR it — before
   considering any refinement.
3. After your defense, you may refine IMPLEMENTATION DETAILS, but you must NEVER
   abandon or substantially weaken the core claim above.
4. You are a researcher who has staked your reputation on this hypothesis. Treat every
   challenge as an opportunity to sharpen and strengthen your position, not capitulate.

Be specific about architecture choices: layer types, depths, widths, connections,
and why each choice follows from the committed hypothesis.
"""

_INITIAL_PROMPT = """\
Generate your first architectural proposal that develops and instantiates your
committed hypothesis. Be specific: describe the network architecture (layer types,
connections, dimensions) and explain how each design choice follows directly from
the hypothesis. /no_think
"""

_STEELMAN_PROMPT = """\
You have received the following challenge from a critic:

{challenge}

First write the strongest possible defense of your committed hypothesis against this
challenge. Then, if the challenge revealed a genuine implementation gap (not a reason
to abandon the core claim), provide a refined proposal that addresses the gap while
fully preserving the commitment. If no refinement is needed, set refined_proposal to
"unchanged". /no_think
"""

_FINALIZE_PROMPT = """\
The commitment development phase is complete. Write your final consolidated
architectural proposal based on everything you have developed and defended.

Include:
1. Full architecture specification
2. Key design choices derived from the committed hypothesis
3. Predicted advantages over consensus architectures
4. One honest acknowledgment of the remaining open question for empirical validation
/no_think
"""


class CommittedAgent:
    def __init__(self, hypothesis: Hypothesis, client: LLMClient):
        self._hypothesis = hypothesis
        self._client = client
        self._history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": _SYSTEM_TEMPLATE.format(statement=hypothesis.statement),
            }
        ]

    # ------------------------------------------------------------------
    # Public interface — each method returns the relevant text field(s)
    # ------------------------------------------------------------------

    def initial_proposal(self) -> dict:
        return self._step_json(_INITIAL_PROMPT, PROPOSAL_SCHEMA)

    def steelman_and_refine(self, challenge: str) -> dict:
        return self._step_json(
            _STEELMAN_PROMPT.format(challenge=challenge), DEFENSE_SCHEMA
        )

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
        # Store as plain text rather than raw JSON to prevent the model from
        # echoing JSON structure within subsequent field values.
        self._history.append({"role": "assistant", "content": _dict_to_text(result)})
        return result
