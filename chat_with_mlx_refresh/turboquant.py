from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

from .model_config import ModelConfig, ModelConfigStore


logger = logging.getLogger(__name__)


class TurboQuantAdapter:
    _patched = False

    @classmethod
    def is_enabled(cls, model_config: ModelConfig | None) -> bool:
        if model_config is None:
            return False
        return (
            ModelConfigStore.normalize_kv_cache_backend(model_config.kv_cache_backend)
            == ModelConfigStore.KV_CACHE_BACKEND_TURBOQUANT
        )

    @classmethod
    def build_text_prompt_cache(cls, model: Any, model_config: ModelConfig | None, default_cache: Any) -> Any:
        if not cls.is_enabled(model_config):
            return default_cache

        turbo_module = cls._ensure_attention_patch()
        bits = ModelConfigStore.normalize_turboquant_bits(getattr(model_config, "turboquant_bits", None)) or 4

        replaced_layers = 0
        for layer_index, (layer, cache_entry) in enumerate(zip(getattr(model, "layers", []), default_cache)):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None or not cls._is_supported_attention_cache(cache_entry):
                continue

            head_dim = cls._resolve_head_dim(self_attn)
            if head_dim is None:
                logger.warning("Skipping TurboQuant cache replacement for layer %s because head_dim is unavailable.", layer_index)
                continue

            default_cache[layer_index] = turbo_module.TurboQuantKVCache(
                head_dim=head_dim,
                bits=bits,
                seed=42 + layer_index,
            )
            replaced_layers += 1

        if replaced_layers == 0:
            raise RuntimeError(
                "TurboQuant is enabled for this model, but no compatible attention KV cache layers were found."
            )
        return default_cache

    @classmethod
    def _ensure_attention_patch(cls):
        turbo_module = importlib.import_module("optiq.core.turbo_kv_cache")
        turbo_module.patch_attention()

        if cls._patched:
            return turbo_module

        base_module = importlib.import_module("mlx_lm.models.base")
        patched_attention = getattr(base_module, "scaled_dot_product_attention")
        for module_name, module in tuple(sys.modules.items()):
            if not module_name.startswith("mlx_lm.models."):
                continue
            if hasattr(module, "scaled_dot_product_attention"):
                setattr(module, "scaled_dot_product_attention", patched_attention)

        cls._patched = True
        return turbo_module

    @staticmethod
    def _is_supported_attention_cache(cache_entry: Any) -> bool:
        if cache_entry is None:
            return False
        if cache_entry.__class__.__name__ == "ArraysCache":
            return False
        return all(hasattr(cache_entry, attr) for attr in ("offset", "update_and_fetch", "make_mask"))

    @staticmethod
    def _resolve_head_dim(self_attn: Any) -> int | None:
        head_dim = getattr(self_attn, "head_dim", None)
        if isinstance(head_dim, int) and head_dim > 0:
            return head_dim
        if hasattr(head_dim, "item"):
            try:
                resolved_head_dim = int(head_dim.item())
            except Exception:
                resolved_head_dim = 0
            if resolved_head_dim > 0:
                return resolved_head_dim

        scale = getattr(self_attn, "scale", None)
        if hasattr(scale, "item"):
            try:
                scale = float(scale.item())
            except Exception:
                scale = None
        elif isinstance(scale, (int, float)):
            scale = float(scale)
        else:
            scale = None

        if scale is None or scale <= 0:
            return None

        inferred_head_dim = round(scale ** -2)
        return inferred_head_dim if inferred_head_dim > 0 else None
