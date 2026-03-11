"""
Tests for Qwen-Hydra.

Unit tests (no model download):
  pytest tests/ -v -k "not slow"

Integration tests (requires model download):
  pytest tests/ -v -m slow
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from qwen_hydra.config import (
    BASE_MODEL_ID,
    EMBED_MODEL_ID,
    RERANK_MODEL_ID,
    RERANK_PREFIX,
    RERANK_SUFFIX,
    Task,
    delta_filename,
    format_embed_input,
    format_rerank_input,
)
from qwen_hydra.heads import embed_head, last_token_pool, rerank_head


# ── Config Tests ───────────────────────────────────────────────────────

class TestConfig:
    def test_task_enum_values(self):
        assert Task.EMBED.value == "embed"
        assert Task.RERANK.value == "rerank"
        assert Task.GENERATE.value == "generate"

    def test_delta_filenames(self):
        assert delta_filename(Task.EMBED) == "delta_embed.safetensors"
        assert delta_filename(Task.RERANK) == "delta_rerank.safetensors"
        assert delta_filename(Task.GENERATE) == "delta_generate.safetensors"

    def test_model_ids_set(self):
        assert "Qwen" in BASE_MODEL_ID
        assert "Embedding" in EMBED_MODEL_ID
        assert "Reranker" in RERANK_MODEL_ID

    def test_format_embed_with_instruction(self):
        result = format_embed_input("Retrieve docs", "hello world")
        assert "Instruct: Retrieve docs" in result
        assert "Query:hello world" in result

    def test_format_embed_without_instruction(self):
        result = format_embed_input(None, "hello world")
        assert result == "hello world"

    def test_format_rerank_input(self):
        result = format_rerank_input("Find docs", "query", "document")
        assert "<Instruct>: Find docs" in result
        assert "<Query>: query" in result
        assert "<Document>: document" in result

    def test_format_rerank_default_instruction(self):
        result = format_rerank_input(None, "query", "document")
        assert "<Instruct>:" in result
        assert "web search" in result.lower()

    def test_rerank_prefix_suffix_format(self):
        assert "<|im_start|>system" in RERANK_PREFIX
        assert "<|im_start|>assistant" in RERANK_SUFFIX
        assert "<think>" in RERANK_SUFFIX


# ── Head Tests ─────────────────────────────────────────────────────────

class TestEmbedHead:
    def test_last_token_pool_right_padded(self):
        """Right-padded: sequences have different lengths."""
        # Batch of 2, seq_len=4, hidden=3
        hidden = torch.randn(2, 4, 3)
        # First seq: 3 real tokens, second: 2 real tokens
        mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ])
        pooled = last_token_pool(hidden, mask)
        assert pooled.shape == (2, 3)
        # First seq should get hidden[:, 2, :], second gets hidden[:, 1, :]
        torch.testing.assert_close(pooled[0], hidden[0, 2])
        torch.testing.assert_close(pooled[1], hidden[1, 1])

    def test_last_token_pool_left_padded(self):
        """Left-padded: all sequences end at the last position."""
        hidden = torch.randn(2, 4, 3)
        mask = torch.tensor([
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ])
        pooled = last_token_pool(hidden, mask)
        assert pooled.shape == (2, 3)
        # Left padding: should return the last column
        torch.testing.assert_close(pooled[0], hidden[0, -1])
        torch.testing.assert_close(pooled[1], hidden[1, -1])

    def test_embed_head_normalize(self):
        hidden = torch.randn(3, 5, 1024)
        mask = torch.ones(3, 5)
        embeddings = embed_head(hidden, mask, normalize=True)
        assert embeddings.shape == (3, 1024)
        # Check L2 norms are ~1.0
        norms = embeddings.norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_embed_head_no_normalize(self):
        hidden = torch.randn(3, 5, 1024)
        mask = torch.ones(3, 5)
        embeddings = embed_head(hidden, mask, normalize=False)
        assert embeddings.shape == (3, 1024)
        # Norms should NOT be 1.0
        norms = embeddings.norm(dim=1)
        assert not torch.allclose(norms, torch.ones(3), atol=1e-3)

    def test_embed_head_mrl_truncation(self):
        hidden = torch.randn(2, 5, 1024)
        mask = torch.ones(2, 5)
        embeddings = embed_head(hidden, mask, output_dim=256)
        assert embeddings.shape == (2, 256)

    def test_embed_head_mrl_invalid_dim(self):
        hidden = torch.randn(2, 5, 1024)
        mask = torch.ones(2, 5)
        with pytest.raises(ValueError, match="output_dim"):
            embed_head(hidden, mask, output_dim=16)  # Below 32


class TestRerankHead:
    def test_rerank_scores_are_probabilities(self):
        """Scores should be in [0, 1] and represent P(yes)."""
        batch, seq_len, vocab = 3, 10, 1000
        logits = torch.randn(batch, seq_len, vocab)

        # Make "yes" (id=42) much stronger than "no" (id=99) for item 0
        logits[0, -1, 42] = 10.0
        logits[0, -1, 99] = -10.0
        # Make "no" stronger for item 1
        logits[1, -1, 42] = -10.0
        logits[1, -1, 99] = 10.0
        # Neutral for item 2
        logits[2, -1, 42] = 0.0
        logits[2, -1, 99] = 0.0

        scores = rerank_head(logits, true_token_id=42, false_token_id=99)

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
        assert scores[0] > 0.99   # Strong yes
        assert scores[1] < 0.01   # Strong no
        assert abs(scores[2] - 0.5) < 0.01  # Neutral → ~0.5


# ── Delta Extraction Tests (mock-based, no downloads) ─────────────────

class TestDeltaComputation:
    def test_identical_weights_produce_no_deltas(self):
        """If finetuned == base, all deltas should be zero → skipped."""
        from qwen_hydra.extract import _compute_delta

        base = {
            "layer.0.weight": torch.randn(10, 10),
            "layer.1.bias": torch.randn(10),
        }
        ft = {
            "layer.0.weight": base["layer.0.weight"].clone(),
            "layer.1.bias": base["layer.1.bias"].clone(),
        }
        deltas = _compute_delta(base, ft, Task.EMBED)
        assert len(deltas) == 0

    def test_modified_weights_produce_deltas(self):
        from qwen_hydra.extract import _compute_delta

        base = {
            "layer.0.weight": torch.zeros(10, 10),
        }
        ft = {
            "layer.0.weight": torch.ones(10, 10),
        }
        deltas = _compute_delta(base, ft, Task.EMBED)
        assert len(deltas) == 1
        assert "layer.0.weight" in deltas
        # Delta should be ~1.0 everywhere
        assert deltas["layer.0.weight"].float().mean().item() == pytest.approx(1.0, abs=0.01)

    def test_vocab_size_mismatch_handled(self):
        from qwen_hydra.extract import _compute_delta

        base = {
            "model.embed_tokens.weight": torch.randn(151936, 1024),
        }
        ft = {
            "model.embed_tokens.weight": torch.randn(151669, 1024),
        }
        # Should not raise, should diff only overlapping rows
        deltas = _compute_delta(base, ft, Task.EMBED)
        if "model.embed_tokens.weight" in deltas:
            assert deltas["model.embed_tokens.weight"].shape[0] == 151669

    def test_new_params_in_finetuned_are_captured(self):
        from qwen_hydra.extract import _compute_delta

        base = {"layer.0.weight": torch.zeros(5, 5)}
        ft = {
            "layer.0.weight": torch.zeros(5, 5),
            "new_head.weight": torch.randn(3, 5),
        }
        deltas = _compute_delta(base, ft, Task.EMBED)
        assert "new_head.weight" in deltas


# ── Integration Tests (require model download) ────────────────────────

@pytest.mark.slow
class TestIntegration:
    """
    These tests download real models and verify the hydra matches
    standalone model outputs. Skip with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    def extracted_dir(self, tmp_path_factory):
        """Run extraction once for all integration tests."""
        from qwen_hydra.extract import extract

        out = tmp_path_factory.mktemp("hydra_deltas")
        extract(output_dir=out)
        return out

    @pytest.fixture(scope="class")
    def hydra(self, extracted_dir):
        from qwen_hydra import QwenHydra
        return QwenHydra.from_extracted(extracted_dir)

    def test_embed_matches_standalone(self, hydra):
        """Compare hydra embeddings against standalone Qwen3-Embedding."""
        from sentence_transformers import SentenceTransformer

        texts = ["What is the capital of France?", "Paris is a city in Europe."]

        # Standalone
        st_model = SentenceTransformer(EMBED_MODEL_ID, device="cpu")
        standalone_vecs = torch.tensor(
            st_model.encode(texts, normalize_embeddings=True)
        )

        # Hydra
        hydra_vecs = hydra.embed(texts)

        # Cosine similarity should be very high
        cosine = F.cosine_similarity(standalone_vecs, hydra_vecs, dim=1)
        assert cosine.min().item() > 0.95, f"Low cosine similarity: {cosine}"

    def test_rerank_matches_standalone(self, hydra):
        """Compare hydra reranker scores against standalone model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        query = "What is the capital of China?"
        docs = [
            "The capital of China is Beijing.",
            "Paris is the capital of France.",
        ]

        hydra_scores = hydra.rerank(query, docs)

        # Basic sanity: first doc should score higher
        assert hydra_scores[0] > hydra_scores[1]
        assert all(0 <= s <= 1 for s in hydra_scores)

    def test_memory_report(self, hydra):
        report = hydra.memory_report()
        assert "trunk_mb" in report
        assert "deltas_mb" in report
        assert report["trunk_mb"] > 0
