from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """OpenAI-compatible endpoint settings. Works with Ollama, llama.cpp, or real API."""
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen3.5:9b-64k"
    temperature: float = 0.7
    # None = no output token limit (required for grammar-constrained JSON generation
    # with Ollama — the constrained decoding uses more tokens than free-form text).
    max_tokens: Optional[int] = None


@dataclass
class ExperimentConfig:
    # How many develop-and-defend steps before evaluation phase
    commitment_steps: int = 5
    # "haas" for committed agent, "baseline" for standard generate-critique-refine
    condition: str = "haas"
    # Key from HYPOTHESIS_POOL (e.g. "no_skip"); None picks one at random
    hypothesis_id: Optional[str] = None
    wandb_project: str = "haas"
    wandb_entity: Optional[str] = None
    seed: int = 42
    # Optional human-readable label for the W&B run
    run_name: Optional[str] = None
