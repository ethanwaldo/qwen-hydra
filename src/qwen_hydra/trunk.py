"""
Shared Trunk Manager.

Loads the Qwen3-0.6B base model once and manages in-place delta application
to swap between tasks (embed, rerank, generate) without reloading.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer

from qwen_hydra.config import (
    DELTA_MANIFEST,
    TOKENIZER_DIR,
    TRUNK_CONFIG,
    TRUNK_WEIGHTS,
    Task,
    delta_filename,
)

log = logging.getLogger("qwen_hydra.trunk")


class SharedTrunk:
    """
    Manages a single Qwen3-0.6B model instance and swaps weight deltas
    in-place to serve different tasks.

    The trunk stores a frozen copy of the base weights so it can always
    cleanly reset before applying a new delta set.
    """

    def __init__(
        self,
        extracted_dir: Path,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
    ):
        self.extracted_dir = Path(extracted_dir)
        self.device = device
        self.dtype = dtype or torch.bfloat16
        self._active_task: Optional[Task] = None

        # Load manifest
        manifest_path = self.extracted_dir / DELTA_MANIFEST
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        # Load tokenizer
        tok_dir = self.extracted_dir / TOKENIZER_DIR
        if tok_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(tok_dir), trust_remote_code=True, padding_side="left",
            )
        else:
            log.warning("No tokenizer dir found, loading from HF hub")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.manifest["base_model"],
                trust_remote_code=True,
                padding_side="left",
            )

        # Load the causal LM model (includes LM head for rerank/generate)
        log.info("Loading trunk model from %s ...", self.extracted_dir)
        config_path = self.extracted_dir / TRUNK_CONFIG
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.extracted_dir),
            config=str(config_path) if config_path.exists() else None,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

        # Store frozen copies of base weights for clean delta swaps
        log.info("Freezing base weights for delta swapping...")
        self._base_state: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            self._base_state[name] = param.data.clone()
            param.requires_grad_(False)

        # Pre-load all delta sets into memory (they're tiny)
        self._deltas: dict[Task, dict[str, torch.Tensor]] = {}
        for task in [Task.EMBED, Task.RERANK]:
            task_info = self.manifest["tasks"].get(task.value, {})
            df = task_info.get("delta_file")
            if df:
                delta_path = self.extracted_dir / df
                if delta_path.exists():
                    self._deltas[task] = load_file(
                        str(delta_path), device="cpu"
                    )
                    log.info(
                        "  Loaded %s deltas: %d params",
                        task.value, len(self._deltas[task]),
                    )

        log.info("Trunk ready. Active task: none")

    @property
    def active_task(self) -> Optional[Task]:
        return self._active_task

    def _reset_to_base(self) -> None:
        """Restore all model weights to the frozen base state."""
        state = self.model.state_dict()
        for name, base_tensor in self._base_state.items():
            if name in state:
                state[name].copy_(base_tensor)
        self._active_task = None

    def switch_task(self, task: Task) -> None:
        """
        Switch the trunk to serve a specific task.

        If already on the requested task, this is a no-op.
        Otherwise, resets to base weights and applies the task's deltas.
        """
        if self._active_task == task:
            return

        log.info("Switching trunk: %s → %s", self._active_task, task.value)

        # Reset to base
        self._reset_to_base()

        # Apply deltas (generate uses base weights directly)
        if task in self._deltas:
            deltas = self._deltas[task]
            state = self.model.state_dict()
            applied = 0
            for name, delta in deltas.items():
                if name in state:
                    param = state[name]
                    # Handle vocab-size mismatch: delta may be smaller
                    if delta.shape != param.shape:
                        min_rows = min(delta.shape[0], param.shape[0])
                        param[:min_rows] += delta[:min_rows].to(
                            dtype=param.dtype, device=param.device
                        )
                    else:
                        param.add_(delta.to(dtype=param.dtype, device=param.device))
                    applied += 1
            log.info("  Applied %d/%d deltas", applied, len(deltas))

        self._active_task = task

    def get_base_model(self) -> AutoModelForCausalLM:
        """Return the underlying model (with current deltas applied)."""
        return self.model

    def get_inner_model(self) -> torch.nn.Module:
        """
        Return the inner transformer model (without LM head).

        For embedding, we need `model.model` (the Qwen3Model inside
        Qwen3ForCausalLM) to get `last_hidden_state`.
        """
        return self.model.model

    def memory_report(self) -> dict:
        """Report memory usage breakdown."""
        trunk_bytes = sum(
            p.numel() * p.element_size()
            for p in self.model.parameters()
        )
        base_bytes = sum(
            t.numel() * t.element_size()
            for t in self._base_state.values()
        )
        delta_bytes = sum(
            sum(d.numel() * d.element_size() for d in deltas.values())
            for deltas in self._deltas.values()
        )
        return {
            "trunk_mb": trunk_bytes / 1e6,
            "base_copy_mb": base_bytes / 1e6,
            "deltas_mb": delta_bytes / 1e6,
            "total_mb": (trunk_bytes + base_bytes + delta_bytes) / 1e6,
            "active_task": self._active_task.value if self._active_task else None,
        }
