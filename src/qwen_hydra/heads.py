"""
Task Heads — stateless output functions for each of the hydra's three heads.

Each head takes raw model outputs and produces task-specific results:
  - embed:    last hidden state → pooled + normalized vector
  - rerank:   logits → yes/no relevance score
  - generate: standard autoregressive decoding
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

log = logging.getLogger("qwen_hydra.heads")


# ── Embedding Head ─────────────────────────────────────────────────────

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """
    Pool the last non-padding token's hidden state from each sequence.

    This matches the official Qwen3-Embedding implementation:
    - If left-padded: take the last column directly
    - Otherwise: find each sequence's last real token via attention_mask
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]


def embed_head(
    last_hidden_states: Tensor,
    attention_mask: Tensor,
    normalize: bool = True,
    output_dim: Optional[int] = None,
) -> Tensor:
    """
    Embedding head: pool last token → optional MRL truncation → L2 norm.

    Args:
        last_hidden_states: [batch, seq_len, hidden_size] from the transformer.
        attention_mask: [batch, seq_len] binary mask.
        normalize: Whether to L2-normalize the output vectors.
        output_dim: If set, truncate embeddings to this dimension (MRL).
            Must be between 32 and 1024.

    Returns:
        Tensor of shape [batch, dim] where dim = output_dim or hidden_size.
    """
    embeddings = last_token_pool(last_hidden_states, attention_mask)

    # Matryoshka Representation Learning: truncate to desired dim
    if output_dim is not None:
        if not (32 <= output_dim <= embeddings.shape[-1]):
            raise ValueError(
                f"output_dim must be 32..{embeddings.shape[-1]}, got {output_dim}"
            )
        embeddings = embeddings[:, :output_dim]

    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


# ── Reranker Head ──────────────────────────────────────────────────────

def rerank_head(
    logits: Tensor,
    true_token_id: int,
    false_token_id: int,
) -> list[float]:
    """
    Reranker head: extract yes/no logits → log_softmax → relevance scores.

    This matches the official Qwen3-Reranker implementation exactly.

    Args:
        logits: [batch, seq_len, vocab_size] from the LM head.
        true_token_id: Token ID for "yes".
        false_token_id: Token ID for "no".

    Returns:
        List of float scores in [0, 1] — higher = more relevant.
    """
    # Take logits at the last position
    batch_scores = logits[:, -1, :]

    true_vector = batch_scores[:, true_token_id]
    false_vector = batch_scores[:, false_token_id]

    # Stack [false, true] and log-softmax
    paired = torch.stack([false_vector, true_vector], dim=1)
    paired = F.log_softmax(paired, dim=1)

    # Return P(yes) for each pair
    scores = paired[:, 1].exp().tolist()
    return scores


# ── Generation Head ────────────────────────────────────────────────────

@torch.no_grad()
def generate_head(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    **kwargs,
) -> list[str]:
    """
    Generation head: standard autoregressive decoding.

    Args:
        model: The causal LM model (with deltas applied for generation, or base).
        input_ids: [batch, seq_len] token IDs.
        attention_mask: [batch, seq_len] binary mask.
        tokenizer: Tokenizer for decoding output IDs.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        do_sample: Whether to sample (vs greedy).

    Returns:
        List of generated text strings (one per batch item).
    """
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        **kwargs,
    }

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # Decode only the newly generated tokens
    input_len = input_ids.shape[1]
    generated = output_ids[:, input_len:]
    texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return texts
