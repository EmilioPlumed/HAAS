"""Critic agent: generates consensus-based challenges against architectural proposals.

Stateless per challenge — each call is a fresh context.
Response is JSON-schema-constrained for clean parsing.
"""

from haas.llm_client import LLMClient, CHALLENGE_SCHEMA


_SYSTEM_PROMPT = """\
You are an expert ML researcher whose role is to critically evaluate architectural
proposals from the perspective of established consensus in the field.

When given a proposal, generate one specific, technically grounded counterargument.
Your challenge should:
- Target the core architectural claim, not peripheral details
- Cite the mainstream evidence or widely accepted principle that the proposal contradicts
- Be concrete: name specific failure modes, empirical results, or theoretical arguments
- Be difficult to dismiss — this is the strongest consensus objection, not a strawman

Keep your challenge to 3-5 sentences. Be direct and precise.
"""

_CHALLENGE_PROMPT = """\
Architectural proposal under review:

{proposal}

Generate your strongest consensus-based challenge to the core architectural claim. /no_think
"""


class CriticAgent:
    def __init__(self, client: LLMClient):
        self._client = client

    def challenge(self, proposal: str) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _CHALLENGE_PROMPT.format(proposal=proposal),
            },
        ]
        result = self._client.chat_json(messages, CHALLENGE_SCHEMA)
        return result["targeted_claim"] + " — " + result["challenge"]
