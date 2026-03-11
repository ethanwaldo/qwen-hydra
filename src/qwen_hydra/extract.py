"""
Delta Extraction Engine.

Downloads all three Qwen3-0.6B variants from HuggingFace, diffs each
fine-tuned model against the base, and saves compact weight deltas as
safetensors files.

First run downloads ~3.6 GB of models. The extracted deltas are typically
only a few MB each.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from qwen_hydra.config import (
    BASE_MODEL_ID,
    DELTA_MANIFEST,
    EMBED_RERANK_VOCAB_SIZE,
    TASK_TO_MODEL_ID,
    TOKENIZER_DIR,
    TRUNK_CONFIG,
    TRUNK_WEIGHTS,
    Task,
    delta_filename,
)

log = logging.getLogger("qwen_hydra.extract")


# Parameters whose shapes are expected to differ between base and
# fine-tuned variants due to vocab-size mismatch (base=151936 vs 151669).
# Only the overlapping rows are diffed; the rest are ignored.
_VOCAB_PARAMS = {"model.embed_tokens.weight", "lm_head.weight"}


def _download_model(model_id: str, cache_dir: Optional[Path] = None) -> Path:
    """Download a HuggingFace model snapshot, return local path."""
    log.info("Downloading %s ...", model_id)
    path = snapshot_download(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*"],
    )
    log.info("  → %s", path)
    return Path(path)


def _load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from a model directory into a flat dict."""
    weights: dict[str, torch.Tensor] = {}
    for st_file in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_file(str(st_file), device="cpu"))
    if not weights:
        raise FileNotFoundError(
            f"No .safetensors files found in {model_dir}"
        )
    return weights


def _compute_delta(
    base_weights: dict[str, torch.Tensor],
    finetuned_weights: dict[str, torch.Tensor],
    task: Task,
    sparsity_threshold: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Compute weight deltas: delta = finetuned - base.

    For vocab-sized parameters, only diffs the overlapping rows.
    Skips parameters where the delta is all zeros (within threshold).
    """
    deltas: dict[str, torch.Tensor] = {}
    matched = 0
    skipped_zero = 0
    skipped_missing = 0

    for name, base_tensor in base_weights.items():
        if name not in finetuned_weights:
            skipped_missing += 1
            continue

        ft_tensor = finetuned_weights[name]

        # Handle vocab-size mismatch: only diff overlapping rows
        if name in _VOCAB_PARAMS and base_tensor.shape != ft_tensor.shape:
            min_rows = min(base_tensor.shape[0], ft_tensor.shape[0])
            delta = ft_tensor[:min_rows].to(torch.float32) - base_tensor[:min_rows].to(torch.float32)
        else:
            if base_tensor.shape != ft_tensor.shape:
                log.warning(
                    "  Shape mismatch for %s: base=%s ft=%s — skipping",
                    name, base_tensor.shape, ft_tensor.shape,
                )
                continue
            delta = ft_tensor.to(torch.float32) - base_tensor.to(torch.float32)

        # Skip near-zero deltas to save space
        if delta.abs().max().item() < sparsity_threshold:
            skipped_zero += 1
            continue

        # Store as bf16 to save space (LoRA deltas are small values)
        deltas[name] = delta.to(torch.bfloat16)
        matched += 1

    # Check for params in finetuned but not in base (new heads, etc.)
    for name, ft_tensor in finetuned_weights.items():
        if name not in base_weights:
            log.info("  New param in %s: %s %s", task.value, name, ft_tensor.shape)
            deltas[name] = ft_tensor.to(torch.bfloat16)

    log.info(
        "  %s: %d delta params, %d unchanged, %d missing from ft",
        task.value, matched, skipped_zero, skipped_missing,
    )
    return deltas


def extract(
    output_dir: Path,
    cache_dir: Optional[Path] = None,
    sparsity_threshold: float = 1e-8,
) -> dict:
    """
    Full extraction pipeline:
    1. Download all three models
    2. Load base weights as reference
    3. Diff each fine-tuned variant against base
    4. Save trunk + deltas to output_dir

    Returns a manifest dict with sizes and metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Download models ────────────────────────────────────────────────
    base_dir = _download_model(BASE_MODEL_ID, cache_dir)
    ft_dirs = {}
    for task in [Task.EMBED, Task.RERANK]:
        model_id = TASK_TO_MODEL_ID[task]
        ft_dirs[task] = _download_model(model_id, cache_dir)

    # ── Load base weights ──────────────────────────────────────────────
    log.info("Loading base model weights...")
    base_weights = _load_safetensors(base_dir)
    log.info("  %d parameters loaded", len(base_weights))

    # ── Save trunk (base weights) ──────────────────────────────────────
    trunk_path = output_dir / TRUNK_WEIGHTS
    log.info("Saving trunk weights to %s ...", trunk_path)
    save_file(base_weights, str(trunk_path))
    trunk_size = trunk_path.stat().st_size

    # ── Copy base config ───────────────────────────────────────────────
    import shutil
    base_config = base_dir / "config.json"
    if base_config.exists():
        shutil.copy2(base_config, output_dir / TRUNK_CONFIG)

    # ── Copy tokenizer files ───────────────────────────────────────────
    tok_dir = output_dir / TOKENIZER_DIR
    tok_dir.mkdir(exist_ok=True)
    for tok_file in base_dir.glob("tokenizer*"):
        shutil.copy2(tok_file, tok_dir / tok_file.name)
    # Also copy special_tokens_map if present
    for extra in ["special_tokens_map.json", "vocab.json", "merges.txt"]:
        src = base_dir / extra
        if src.exists():
            shutil.copy2(src, tok_dir / extra)

    # ── Extract deltas ─────────────────────────────────────────────────
    manifest = {
        "version": 1,
        "base_model": BASE_MODEL_ID,
        "trunk_size_bytes": trunk_size,
        "tasks": {},
    }

    for task in [Task.EMBED, Task.RERANK]:
        log.info("Extracting deltas for %s ...", task.value)
        ft_weights = _load_safetensors(ft_dirs[task])
        deltas = _compute_delta(
            base_weights, ft_weights, task,
            sparsity_threshold=sparsity_threshold,
        )

        delta_path = output_dir / delta_filename(task)
        save_file(deltas, str(delta_path))
        delta_size = delta_path.stat().st_size

        manifest["tasks"][task.value] = {
            "model_id": TASK_TO_MODEL_ID[task],
            "delta_file": delta_filename(task),
            "delta_size_bytes": delta_size,
            "num_delta_params": len(deltas),
        }

        log.info(
            "  Saved %s: %d params, %.2f MB",
            delta_filename(task), len(deltas), delta_size / 1e6,
        )

        # Free memory
        del ft_weights, deltas

    # Generation uses base weights directly — no delta needed
    manifest["tasks"][Task.GENERATE.value] = {
        "model_id": BASE_MODEL_ID,
        "delta_file": None,
        "delta_size_bytes": 0,
        "num_delta_params": 0,
    }

    # ── Save manifest ──────────────────────────────────────────────────
    manifest_path = output_dir / DELTA_MANIFEST
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_delta = sum(
        t["delta_size_bytes"] for t in manifest["tasks"].values()
    )
    log.info(
        "Extraction complete. Trunk: %.1f MB, Deltas: %.2f MB, "
        "Total: %.1f MB (vs ~%.1f MB for 3 separate models)",
        trunk_size / 1e6,
        total_delta / 1e6,
        (trunk_size + total_delta) / 1e6,
        trunk_size * 3 / 1e6,
    )

    return manifest
