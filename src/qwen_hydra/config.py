"""
Constants, model IDs, paths, and template strings for Qwen-Hydra.
"""

from pathlib import Path
from enum import Enum


class Task(str, Enum):
    """The three heads of the hydra."""
    EMBED = "embed"
    RERANK = "rerank"
    GENERATE = "generate"


# ── HuggingFace model IDs ──────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
EMBED_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"

TASK_TO_MODEL_ID = {
    Task.EMBED: EMBED_MODEL_ID,
    Task.RERANK: RERANK_MODEL_ID,
    Task.GENERATE: BASE_MODEL_ID,
}

# ── Architecture constants ─────────────────────────────────────────────
# Shared across all three variants
NUM_LAYERS = 28
HIDDEN_SIZE = 1024
NUM_ATTENTION_HEADS = 16
NUM_KV_HEADS = 8
INTERMEDIATE_SIZE = 3072

# Vocab sizes differ: base=151936, embed/rerank=151669
BASE_VOCAB_SIZE = 151936
EMBED_RERANK_VOCAB_SIZE = 151669

# ── Reranker token IDs ─────────────────────────────────────────────────
# These are looked up dynamically from the tokenizer at init time, but we
# store the token strings here for reference.
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
