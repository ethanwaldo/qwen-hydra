"""
QwenHydra — the unified API.

One trunk, three heads. Switch tasks by swapping tiny weight deltas.

    hydra = QwenHydra.from_extracted("./deltas")
    vecs   = hydra.embed(["text"])
    scores = hydra.rerank("query", ["doc1", "doc2"])
    text   = hydra.generate("prompt", max_new_tokens=128)
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Union

import torch

from qwen_hydra.config import (
    RERANK_PREFIX,
    RERANK_SUFFIX,
    Task,
    format_embed_input,
    format_rerank_input,
    RERANK_TRUE_TOKEN,
    RERANK_FALSE_TOKEN,
)
from qwen_hydra.heads import embed_head, generate_head, rerank_head
from qwen_hydra.trunk import SharedTrunk

log = logging.getLogger("qwen_hydra.hydra")


class QwenHydra:
    """
    Shared-trunk Qwen3-0.6B with three task heads.

    Manages a single model instance and swaps weight deltas on demand
    to serve embedding, reranking, and generation from ~1.3 GB total.
    """

    def __init__(self, trunk: SharedTrunk):
        self._trunk = trunk
        self._lock = threading.Lock()

        # Cache reranker token IDs
        self._true_id = trunk.tokenizer.convert_tokens_to_ids(RERANK_TRUE_TOKEN)
        self._false_id = trunk.tokenizer.convert_tokens_to_ids(RERANK_FALSE_TOKEN)

        # Pre-encode reranker prefix/suffix tokens
        self._rerank_prefix_ids = trunk.tokenizer.encode(
            RERANK_PREFIX, add_special_tokens=False
        )
        self._rerank_suffix_ids = trunk.tokenizer.encode(
            RERANK_SUFFIX, add_special_tokens=False
        )

    @classmethod
    def from_extracted(
        cls,
        extracted_dir: Union[str, Path],
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ) -> "QwenHydra":
        """
        Load a QwenHydra from a directory created by `qwen-hydra extract`.

        Args:
            extracted_dir: Path to the directory containing trunk.safetensors,
                delta_*.safetensors, and manifest.json.
            device: Device to load the model on ("cpu", "cuda", etc).
            dtype: Model dtype (default: bfloat16).
        """
        trunk = SharedTrunk(
            extracted_dir=Path(extracted_dir),
            device=device,
            dtype=dtype,
        )
        return cls(trunk)

    # ── Embedding ──────────────────────────────────────────────────────

    @torch.no_grad()
    def embed(
        self,
        texts: list[str],
        instruction: Optional[str] = None,
        max_length: int = 8192,
        output_dim: Optional[int] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """
        Produce embeddings for a list of texts.

        Args:
            texts: Input texts to embed.
            instruction: Optional task instruction (prepended to each text).
                Use for queries; omit for documents.
            max_length: Max token length per text.
            output_dim: MRL truncation dimension (32–1024). None = full 1024.
            normalize: L2-normalize output vectors.
            batch_size: Processing batch size.

        Returns:
            Tensor of shape [len(texts), dim].
        """
        with self._lock:
            self._trunk.switch_task(Task.EMBED)

        # Format inputs with optional instruction
        formatted = [format_embed_input(instruction, t) for t in texts]

        all_embeddings = []
        for start in range(0, len(formatted), batch_size):
            batch_texts = formatted[start : start + batch_size]

            batch_dict = self._trunk.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {
                k: v.to(self._trunk.device)
                for k, v in batch_dict.items()
            }

            # Use the inner model (without LM head) for hidden states
            inner_model = self._trunk.get_inner_model()
            outputs = inner_model(**batch_dict)

            embeddings = embed_head(
                outputs.last_hidden_state,
                batch_dict["attention_mask"],
                normalize=normalize,
                output_dim=output_dim,
            )
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    # ── Reranking ──────────────────────────────────────────────────────

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: list[str],
        instruction: Optional[str] = None,
        max_length: int = 8192,
    ) -> list[float]:
        """
        Score query-document pairs for relevance.

        Args:
            query: The search query.
            documents: Candidate documents to score.
            instruction: Optional task instruction.
            max_length: Max token length per pair.

        Returns:
            List of float scores in [0, 1], one per document.
            Higher = more relevant.
        """
        with self._lock:
            self._trunk.switch_task(Task.RERANK)

        # Format each query-document pair
        pairs = [
            format_rerank_input(instruction, query, doc)
            for doc in documents
        ]

        # Tokenize with reranker prefix/suffix wrapping
        tokenizer = self._trunk.tokenizer
        inputs = tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=max_length - len(self._rerank_prefix_ids) - len(self._rerank_suffix_ids),
        )

        # Wrap each with prefix/suffix token IDs
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = (
                self._rerank_prefix_ids + ids + self._rerank_suffix_ids
            )

        # Pad to uniform length
        inputs = tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {
            k: v.to(self._trunk.device)
            for k, v in inputs.items()
        }

        # Forward through the full causal LM (need logits)
        model = self._trunk.get_base_model()
        outputs = model(**inputs)

        return rerank_head(outputs.logits, self._true_id, self._false_id)

    # ── Generation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User message.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to sample (vs greedy).
            system_prompt: Optional system message.

        Returns:
            Generated text string.
        """
        with self._lock:
            self._trunk.switch_task(Task.GENERATE)

        # Build chat-format messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        tokenizer = self._trunk.tokenizer
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self._trunk.device) for k, v in inputs.items()}

        model = self._trunk.get_base_model()
        results = generate_head(
            model=model,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs,
        )
        return results[0]

    # ── Utilities ──────────────────────────────────────────────────────

    @property
    def active_task(self) -> Optional[str]:
        """Currently active task name, or None."""
        t = self._trunk.active_task
        return t.value if t else None

    def memory_report(self) -> dict:
        """Get memory usage breakdown."""
        return self._trunk.memory_report()
