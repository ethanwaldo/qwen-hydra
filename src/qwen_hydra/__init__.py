"""
Qwen-Hydra: shared-trunk Qwen3-0.6B with delta swapping.

One trunk, three heads — embed, rerank, generate.
"""

from qwen_hydra.hydra import QwenHydra

__all__ = ["QwenHydra"]
__version__ = "0.1.0"
