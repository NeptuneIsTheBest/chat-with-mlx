from __future__ import annotations

import copy
import gc
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

import mlx

from ..model import BaseLocalModel, MultimodalModel, TextModel


logger = logging.getLogger(__name__)


def create_empty_prompt_cache_state() -> dict[str, Any]:
    return {
        "model_signature": "",
        "backend": "",
        "token_ids": [],
        "media_signature": "",
        "media_sequence": [],
        "prompt_cache": None,
        "status": "invalid",
    }


def delete_prompt_cache_state(state: Any) -> None:
    if isinstance(state, dict):
        state["prompt_cache"] = None
        state["token_ids"] = []
        state["media_sequence"] = []
    gc.collect()
    mlx.core.clear_cache()


@dataclass(slots=True)
class PromptCachePlan:
    backend: str
    prompt: str
    prompt_token_ids: list[int]
    prompt_cache: Any
    cached_prefix_len: int
    media_signature: str
    media_sequence: list[tuple[tuple[str, ...], tuple[str, ...]]]
    prepared_inputs: Optional[dict[str, Any]] = None


@dataclass(slots=True)
class PromptTokenRecorder:
    token_ids: list[int]
    last_generation_tokens: int

    def __init__(self) -> None:
        self.token_ids = []
        self.last_generation_tokens = 0

    def observe(self, chunk: Any) -> None:
        token = getattr(chunk, "token", None)
        if token is None:
            return

        generation_tokens = getattr(chunk, "generation_tokens", None)
        if isinstance(generation_tokens, int):
            if generation_tokens <= self.last_generation_tokens:
                return
            self.last_generation_tokens = generation_tokens

        self.token_ids.append(int(token))


class PromptCacheService:
    def reset_state(self) -> dict[str, Any]:
        return create_empty_prompt_cache_state()

    def prepare_chat_generation(
        self,
        model: BaseLocalModel,
        prompt: str,
        turn_media: list[dict[str, list[str]]],
        state: Optional[dict[str, Any]],
    ) -> Optional[PromptCachePlan]:
        if isinstance(model, TextModel):
            return self._prepare_text_generation(model, prompt, state)
        if isinstance(model, MultimodalModel):
            return self._prepare_multimodal_generation(model, prompt, turn_media, state)
        return None

    def build_success_state(
        self,
        model: BaseLocalModel,
        plan: Optional[PromptCachePlan],
        generated_token_ids: list[int],
    ) -> dict[str, Any]:
        if plan is None:
            return create_empty_prompt_cache_state()

        return {
            "model_signature": self._get_model_signature(model),
            "backend": plan.backend,
            "token_ids": plan.prompt_token_ids + list(generated_token_ids),
            "media_signature": plan.media_signature,
            "media_sequence": [
                {
                    "images": list(images),
                    "audios": list(audios),
                }
                for images, audios in plan.media_sequence
            ],
            "prompt_cache": plan.prompt_cache,
            "status": "ready",
        }

    def _prepare_text_generation(
        self,
        model: TextModel,
        prompt: str,
        state: Optional[dict[str, Any]],
    ) -> PromptCachePlan:
        prompt_token_ids = model.encode_prompt_tokens(prompt)
        prompt_cache = model.make_prompt_cache()
        cached_prefix_len = 0

        reusable_state = self._normalize_state(state)
        if self._can_reuse_text_cache(model, reusable_state, prompt_token_ids):
            cached = self._copy_prompt_cache(reusable_state.get("prompt_cache"))
            if cached is not None:
                prompt_cache = cached
                cached_prefix_len = len(reusable_state["token_ids"])

        return PromptCachePlan(
            backend="text",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_cache=prompt_cache,
            cached_prefix_len=cached_prefix_len,
            media_signature="",
            media_sequence=[],
        )

    def _prepare_multimodal_generation(
        self,
        model: MultimodalModel,
        prompt: str,
        turn_media: list[dict[str, list[str]]],
        state: Optional[dict[str, Any]],
    ) -> PromptCachePlan:
        images, audios = self._flatten_turn_media(turn_media)
        prepared_inputs = model.prepare_prompt_inputs(prompt, images=images, audios=audios)
        prompt_token_ids = model.extract_prompt_token_ids(prepared_inputs)
        media_sequence = self._normalize_turn_media(turn_media)
        media_signature = self._compute_media_signature(turn_media)
        prompt_cache = model.make_prompt_cache()
        cached_prefix_len = 0

        reusable_state = self._normalize_state(state)
        if self._can_reuse_multimodal_cache(model, reusable_state, prompt_token_ids, media_sequence):
            cached = self._copy_prompt_cache(reusable_state.get("prompt_cache"))
            if cached is not None:
                prompt_cache = cached
                cached_prefix_len = len(reusable_state["token_ids"])

        return PromptCachePlan(
            backend="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_cache=prompt_cache,
            cached_prefix_len=cached_prefix_len,
            media_signature=media_signature,
            media_sequence=media_sequence,
            prepared_inputs=prepared_inputs,
        )

    @staticmethod
    def _flatten_turn_media(turn_media: list[dict[str, list[str]]]) -> tuple[list[str], list[str]]:
        images = [path for item in turn_media for path in item.get("images", [])]
        audios = [path for item in turn_media for path in item.get("audios", [])]
        return images, audios

    @staticmethod
    def _normalize_turn_media(turn_media: list[dict[str, list[str]]]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        normalized: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
        for item in turn_media:
            if isinstance(item, tuple) and len(item) == 2:
                raw_images, raw_audios = item
                images = tuple(str(path) for path in raw_images)
                audios = tuple(str(path) for path in raw_audios)
            else:
                images = tuple(str(path) for path in item.get("images", []))
                audios = tuple(str(path) for path in item.get("audios", []))
            normalized.append((images, audios))
        return normalized

    def _normalize_state(self, state: Optional[dict[str, Any]]) -> dict[str, Any]:
        normalized = create_empty_prompt_cache_state()
        if not isinstance(state, dict):
            return normalized

        normalized.update(
            {
                "model_signature": str(state.get("model_signature") or ""),
                "backend": str(state.get("backend") or ""),
                "token_ids": [int(token) for token in list(state.get("token_ids") or [])],
                "media_signature": str(state.get("media_signature") or ""),
                "media_sequence": self._normalize_turn_media(list(state.get("media_sequence") or [])),
                "prompt_cache": state.get("prompt_cache"),
                "status": str(state.get("status") or "invalid"),
            }
        )
        return normalized

    def _can_reuse_text_cache(
        self,
        model: TextModel,
        state: dict[str, Any],
        prompt_token_ids: list[int],
    ) -> bool:
        return (
            state.get("status") == "ready"
            and state.get("backend") == "text"
            and state.get("model_signature") == self._get_model_signature(model)
            and bool(state.get("prompt_cache"))
            and self._is_prefix(state.get("token_ids", []), prompt_token_ids)
        )

    def _can_reuse_multimodal_cache(
        self,
        model: MultimodalModel,
        state: dict[str, Any],
        prompt_token_ids: list[int],
        media_sequence: list[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> bool:
        return (
            state.get("status") == "ready"
            and state.get("backend") == "multimodal"
            and state.get("model_signature") == self._get_model_signature(model)
            and bool(state.get("prompt_cache"))
            and self._is_prefix(list(state.get("media_sequence", [])), media_sequence)
            and self._is_prefix(state.get("token_ids", []), prompt_token_ids)
        )

    @staticmethod
    def _is_prefix(prefix: list[int], values: list[int]) -> bool:
        return bool(prefix) and len(prefix) < len(values) and values[: len(prefix)] == prefix

    @staticmethod
    def _copy_prompt_cache(prompt_cache: Any) -> Any:
        if prompt_cache is None:
            return None
        try:
            return copy.deepcopy(prompt_cache)
        except Exception as exc:
            logger.warning("Prompt cache copy failed, falling back to a cache miss: %s", exc)
            return None

    @staticmethod
    def _get_model_signature(model: BaseLocalModel) -> str:
        return f"{type(model).__name__}:{getattr(model, 'model_path', '')}"

    @staticmethod
    def _compute_media_signature(turn_media: list[dict[str, list[str]]]) -> str:
        parts: list[str] = []
        for item in turn_media:
            images = [str(path) for path in item.get("images", [])]
            audios = [str(path) for path in item.get("audios", [])]
            parts.append(f"images={'|'.join(images)}\naudios={'|'.join(audios)}")
        return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()
