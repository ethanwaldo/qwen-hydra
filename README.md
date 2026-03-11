# Qwen-Hydra

**One trunk, three heads.** Load a single Qwen3 base model and hot-swap weight deltas for embedding, reranking, and generation — instead of loading three separate copies.

## Models

Qwen-Hydra works with the Qwen3 model family. Each size has three variants that share the same transformer architecture — they only differ by small LoRA-scale weight deltas from fine-tuning:

| Size | Base (Generation) | Embedding | Reranker |
|---|---|---|---|
| **0.6B** | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| **4B** | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
| **8B** | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | [Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | [Qwen/Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) |

## How It Works

Qwen's embedding and reranker models are LoRA fine-tunes of the same base model. Normally you'd load all three separately — tripling your memory usage. Qwen-Hydra instead:

1. **Extracts** the tiny weight deltas between each fine-tuned model and the base
2. **Loads** the base model once as a shared trunk
3. **Swaps** deltas in-place when you switch between tasks

The deltas are typically just a few MB each, while the base model is hundreds of MB to several GB.

### Memory Savings

Loading three separate models triples your memory. Qwen-Hydra loads **one** and adds tiny deltas:

| Size | Single Model (bf16) | 3× Separate | Hydra (1 trunk + deltas) |
|---|---|---|---|
| 0.6B | ~1.2 GB | ~3.6 GB | **~1.2 GB** |
| 4B | ~8 GB | ~24 GB | **~8 GB** |
| 8B | ~16 GB | ~48 GB | **~16 GB** |

The deltas are typically just a few MB each — negligible compared to the trunk.

## Quick Start

```bash
pip install -e .
```

### Step 1: Extract Deltas (one-time)

Download the three HuggingFace models for your chosen size, diff them, and save the trunk + deltas:

```bash
# 0.6B (default, ~3.6 GB download)
qwen-hydra extract --output ./deltas-0.6B

# 4B (~24 GB download)
qwen-hydra extract --output ./deltas-4B --size 4B

# 8B (~48 GB download)
qwen-hydra extract --output ./deltas-8B --size 8B
```

### Step 2: Use the Hydra

```python
from qwen_hydra import QwenHydra

hydra = QwenHydra.from_extracted("./deltas-0.6B")

# Embed text into vectors (for search, retrieval, RAG)
vectors = hydra.embed(
    ["What is the capital of France?", "Paris is a city in Europe."],
    instruction="Given a web search query, retrieve relevant passages",
)

# Rerank documents by relevance to a query
scores = hydra.rerank(
    query="What is the capital of France?",
    documents=["Paris is the capital of France.", "Berlin is in Germany."],
)

# Generate text (standard chat completion)
response = hydra.generate("Tell me about Paris", max_new_tokens=256)
print(response)
```

## Architecture

```
┌─────────────────────────────────────┐
│     Qwen3 Base Model (trunk)        │
│  Loaded once into memory            │
│  0.6B / 4B / 8B                     │
├─────────────────────────────────────┤
│  Delta Swap Layer                   │
│  ┌───────────┬────────────┬───────┐ │
│  │ Embed Δ   │ Rerank Δ   │ Gen Δ │ │
│  │ ~few MB   │ ~few MB    │ 0 MB  │ │
│  └───────────┴────────────┴───────┘ │
├─────────────────────────────────────┤
│  Task Heads                         │
│  ┌───────────┬────────────┬───────┐ │
│  │ Pool + L2 │ Yes/No     │ Next  │ │
│  │ Normalize │ Logits     │ Token │ │
│  └───────────┴────────────┴───────┘ │
└─────────────────────────────────────┘
```

- **Embedding head**: Pools the last token's hidden state, L2-normalizes. Supports Matryoshka (MRL) dimension truncation (32–1024). Instruction-aware.
- **Reranker head**: Cross-encoder format. Extracts yes/no logits via log-softmax for a relevance score in [0, 1].
- **Generation head**: Standard autoregressive next-token decoding with chat template support.

## CLI Reference

```bash
# Extract deltas for a model size
qwen-hydra extract --output ./deltas --size 0.6B

# Inspect an extracted delta directory
qwen-hydra info --dir ./deltas
```

## License

MIT
