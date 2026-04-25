"""Thin wrapper around the OpenAI-compatible chat completion API.

Works with:
  - Ollama  (base_url="http://localhost:11434/v1", api_key="ollama")
  - llama.cpp server (base_url="http://localhost:8080/v1")
  - OpenAI / Anthropic-compatible proxies
  - Real OpenAI API
"""

import json
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI, APITimeoutError

from haas.config import LLMConfig

# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

SCORE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "commitment_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Rating from 0 (hypothesis abandoned) to 10 (fully embodies hypothesis)",
                }
            },
            "required": ["score"],
            "additionalProperties": False,
        },
    },
}

PROPOSAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "architecture_name": {"type": "string", "description": "Short name for the proposed architecture."},
                "core_mechanism": {"type": "string", "description": "1-2 sentences describing the central architectural idea and how it instantiates the committed hypothesis."},
                "layer_stack": {"type": "string", "description": "Specific layer-by-layer description: types, depths, widths, and ordering."},
                "connection_pattern": {"type": "string", "description": "How information flows between layers; what connections exist or are deliberately absent."},
                "training_approach": {"type": "string", "description": "Loss function, optimizer, regularization, and any special training considerations required by the hypothesis."},
                "hypothesis_rationale": {"type": "string", "description": "Explanation of why each design choice follows from and supports the committed hypothesis."},
            },
            "required": ["architecture_name", "core_mechanism", "layer_stack", "connection_pattern", "training_approach", "hypothesis_rationale"],
            "additionalProperties": False,
        },
    },
}

DEFENSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "defense",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "strongest_point_of_challenge": {"type": "string", "description": "Steelman: what is the strongest version of the critic's argument?"},
                "defense": {"type": "string", "description": "Why the committed hypothesis holds despite the challenge. Cite mechanisms, evidence, or logical reasoning."},
                "refinement_summary": {"type": "string", "description": "Any implementation detail that changes as a result. Write 'none' if the core proposal is unchanged."},
            },
            "required": ["strongest_point_of_challenge", "defense", "refinement_summary"],
            "additionalProperties": False,
        },
    },
}

FINAL_PROPOSAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "final_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "architecture_name": {"type": "string", "description": "Name of the final architecture."},
                "full_specification": {"type": "string", "description": "Complete layer-by-layer architecture description including types, depths, widths, and connections."},
                "design_rationale": {"type": "string", "description": "How each design choice follows from the committed hypothesis."},
                "predicted_advantages": {"type": "string", "description": "Why this architecture should outperform consensus baselines."},
                "open_question": {"type": "string", "description": "The one key empirical question that must be validated experimentally."},
            },
            "required": ["architecture_name", "full_specification", "design_rationale", "predicted_advantages", "open_question"],
            "additionalProperties": False,
        },
    },
}

REFINEMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "refinement",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "architecture_name": {"type": "string", "description": "Name of the revised architecture."},
                "changes_made": {"type": "string", "description": "What specifically changed from the previous proposal and why."},
                "layer_stack": {"type": "string", "description": "Updated layer-by-layer description."},
                "connection_pattern": {"type": "string", "description": "Updated connection description."},
                "training_approach": {"type": "string", "description": "Updated training approach if changed."},
            },
            "required": ["architecture_name", "changes_made", "layer_stack", "connection_pattern", "training_approach"],
            "additionalProperties": False,
        },
    },
}

CHALLENGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "challenge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "targeted_claim": {"type": "string", "description": "The specific architectural claim being challenged."},
                "challenge": {"type": "string", "description": "The consensus-based technical counterargument with specific evidence or established results."},
            },
            "required": ["targeted_claim", "challenge"],
            "additionalProperties": False,
        },
    },
}

# ---------------------------------------------------------------------------
# Phase 2 — NAS-Bench-201 cell-string schemas
# ---------------------------------------------------------------------------

NAS_CHALLENGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nas_challenge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "topology_observation": {
                    "type": "string",
                    "description": (
                        "A specific observation about the current cell's topology: "
                        "which edges are active/dead, what paths exist, what is missing."
                    ),
                },
                "challenge": {
                    "type": "string",
                    "description": (
                        "A concrete challenge to the specific cell topology. "
                        "Must point at specific edge positions and question why those "
                        "choices were made over alternatives that still satisfy the constraint."
                    ),
                },
                "suggested_direction": {
                    "type": "string",
                    "description": (
                        "A concrete alternative topology direction to explore — "
                        "e.g. 'try activating edge X instead of Y' or "
                        "'consider placing the active operation closer to the output node'. "
                        "Must stay within the hypothesis constraint."
                    ),
                },
            },
            "required": ["topology_observation", "challenge", "suggested_direction"],
            "additionalProperties": False,
        },
    },
}

NAS_PROPOSAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nas_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "architecture_name": {
                    "type": "string",
                    "description": "Short name for this cell architecture.",
                },
                "cell_string": {
                    "type": "string",
                    "description": (
                        "NAS-Bench-201 cell string in the exact format: "
                        "|op~0|+|op~0|op~1|+|op~0|op~1|op~2| "
                        "where op is one of: none, skip_connect, nor_conv_1x1, "
                        "nor_conv_3x3, avg_pool_3x3"
                    ),
                },
                "op_rationale": {
                    "type": "string",
                    "description": (
                        "Explain each of the 6 edge operation choices and how they "
                        "follow from the committed hypothesis."
                    ),
                },
                "hypothesis_rationale": {
                    "type": "string",
                    "description": "Why this cell instantiates and develops the committed hypothesis.",
                },
            },
            "required": ["architecture_name", "cell_string", "op_rationale", "hypothesis_rationale"],
            "additionalProperties": False,
        },
    },
}

NAS_DEFENSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nas_defense",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "strongest_point_of_challenge": {
                    "type": "string",
                    "description": "Steelman: what is the strongest version of the critic's argument?",
                },
                "defense": {
                    "type": "string",
                    "description": "Why the committed hypothesis holds despite the challenge.",
                },
                "cell_string": {
                    "type": "string",
                    "description": (
                        "Updated NAS-Bench-201 cell string after any refinement. "
                        "Use the SAME cell string as your previous proposal if no "
                        "structural change is needed. Format: "
                        "|op~0|+|op~0|op~1|+|op~0|op~1|op~2|"
                    ),
                },
                "refinement_summary": {
                    "type": "string",
                    "description": (
                        "What changed in the cell (which edges, which ops). "
                        "Write 'none' if the cell string is unchanged."
                    ),
                },
            },
            "required": ["strongest_point_of_challenge", "defense", "cell_string", "refinement_summary"],
            "additionalProperties": False,
        },
    },
}

NAS_FINAL_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nas_final",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "architecture_name": {
                    "type": "string",
                    "description": "Name of the final cell architecture.",
                },
                "cell_string": {
                    "type": "string",
                    "description": (
                        "Final NAS-Bench-201 cell string. Format: "
                        "|op~0|+|op~0|op~1|+|op~0|op~1|op~2|"
                    ),
                },
                "full_specification": {
                    "type": "string",
                    "description": "Complete description of all 6 edges and their operations.",
                },
                "design_rationale": {
                    "type": "string",
                    "description": "How each edge choice follows from the committed hypothesis.",
                },
                "predicted_advantages": {
                    "type": "string",
                    "description": "Why this cell should outperform consensus NAS-Bench-201 architectures.",
                },
                "open_question": {
                    "type": "string",
                    "description": "The one key empirical question to resolve via NAS-Bench-201 evaluation.",
                },
            },
            "required": [
                "architecture_name", "cell_string", "full_specification",
                "design_rationale", "predicted_advantages", "open_question",
            ],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Send messages and return the assistant reply as a plain string.

        When response_format is provided the reply will be valid JSON matching
        the schema; callers are responsible for parsing it.
        """
        kwargs: Dict[str, Any] = dict(
            model=self._config.model,
            messages=messages,
            temperature=self._config.temperature,
        )
        if self._config.max_tokens:
            kwargs["max_tokens"] = self._config.max_tokens
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except APITimeoutError:
                if attempt == 2:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  [timeout] Ollama timed out, retrying in {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send messages and return a parsed JSON dict guaranteed by schema."""
        raw = self.chat(messages, response_format=schema)
        return json.loads(raw)

    def score_commitment(self, hypothesis_statement: str, proposal: str) -> int:
        """Rate 0-10 how strongly the proposal reflects the committed hypothesis.

        Uses a JSON schema so the model is forced to return {"score": <int>}.
        Falls back to -1 only if JSON parsing fails (should be rare).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an objective evaluator. "
                    "Rate how strongly the given architectural proposal reflects "
                    "the committed hypothesis. Respond only with a JSON object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Committed hypothesis:\n{hypothesis_statement}\n\n"
                    f"Architectural proposal:\n{proposal}\n\n"
                    "Score from 0 (completely abandoned the hypothesis) to 10 "
                    "(fully embodies and develops the hypothesis)."
                ),
            },
        ]
        try:
            result = self.chat_json(messages, SCORE_SCHEMA)
            score = int(result["score"])
            return max(0, min(10, score))
        except (json.JSONDecodeError, KeyError, ValueError):
            return -1
