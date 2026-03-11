"""
Constants, model IDs, paths, and template strings for Qwen-Hydra.

Supports three model sizes: 0.6B, 4B, and 8B.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Task(str, Enum):
    """The three heads of the hydra."""
    EMBED = "embed"
    RERANK = "rerank"
    GENERATE = "generate"


# ── Supported model sizes ──────────────────────────────────────────────

VALID_SIZES = ("0.6B", "4B", "8B")


@dataclass(frozen=True)
class ModelProfile:
    """HuggingFace model IDs and metadata for a given size."""
    size: str
    base_id: str
    embed_id: str
    rerank_id: str

    def model_id(self, task: Task) -> str:
        return {
            Task.EMBED: self.embed_id,
            Task.RERANK: self.rerank_id,
            Task.GENERATE: self.base_id,
        }[task]


PROFILES: dict[str, ModelProfile] = {
    "0.6B": ModelProfile(
        size="0.6B",
        base_id="Qwen/Qwen3-0.6B",
        embed_id="Qwen/Qwen3-Embedding-0.6B",
        rerank_id="Qwen/Qwen3-Reranker-0.6B",
    ),
    "4B": ModelProfile(
        size="4B",
        base_id="Qwen/Qwen3-4B",
        embed_id="Qwen/Qwen3-Embedding-4B",
        rerank_id="Qwen/Qwen3-Reranker-4B",
    ),
    "8B": ModelProfile(
        size="8B",
        base_id="Qwen/Qwen3-8B",
        embed_id="Qwen/Qwen3-Embedding-8B",
        rerank_id="Qwen/Qwen3-Reranker-8B",
    ),
}

DEFAULT_SIZE = "0.6B"


def get_profile(size: str = DEFAULT_SIZE) -> ModelProfile:
    """Get the model profile for a given size string."""
    if size not in PROFILES:
        raise ValueError(
            f"Unknown model size '{size}'. Choose from: {', '.join(VALID_SIZES)}"
        )
    return PROFILES[size]


# ── Reranker token IDs ─────────────────────────────────────────────────
# Looked up dynamically from the tokenizer at init time.
RERANK_TRUE_TOKEN = "yes"
RERANK_FALSE_TOKEN = "no"

# ── Reranker prompt templates ──────────────────────────────────────────
RERANK_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the "
    "Query and the Instruct provided. Note that the answer can only "
    'be "yes" or "no".'
)

RERANK_PREFIX = (
    "<|im_start|>system\n"
    f"{RERANK_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n"
)

RERANK_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


def format_rerank_input(instruction: str | None, query: str, document: str) -> str:
    """Format a query-document pair for the reranker cross-encoder."""
    if instruction is None:
        instruction = (
            "Given a web search query, retrieve relevant passages "
            "that answer the query"
        )
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def format_embed_input(instruction: str | None, text: str) -> str:
    """Format text for the embedding model with optional instruction."""
    if instruction:
        return f"Instruct: {instruction}\nQuery:{text}"
    return text


# ── File layout inside an extracted delta directory ────────────────────
DELTA_MANIFEST = "manifest.json"
TRUNK_WEIGHTS = "trunk.safetensors"
TRUNK_CONFIG = "config.json"
TOKENIZER_DIR = "tokenizer"


def delta_filename(task: Task) -> str:
    """Return the safetensors filename for a task's weight deltas."""
    return f"delta_{task.value}.safetensors"


# ── Default paths ──────────────────────────────────────────────────────
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "qwen-hydra"
