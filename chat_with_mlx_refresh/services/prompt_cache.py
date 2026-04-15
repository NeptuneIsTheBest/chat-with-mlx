from __future__ import annotations

import copy
import gc
import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import mlx


logger = logging.getLogger(__name__)

PROMPT_CACHE_STATE_EMPTY = "empty"
PROMPT_CACHE_STATE_READY = "ready"
VALID_PROMPT_CACHE_STATES = frozenset({PROMPT_CACHE_STATE_EMPTY, PROMPT_CACHE_STATE_READY})


def create_prompt_cache_state(
    cache_session_id: str = "",
    status: str = PROMPT_CACHE_STATE_EMPTY,
) -> dict[str, Any]:
    normalized_status = status if status in VALID_PROMPT_CACHE_STATES else PROMPT_CACHE_STATE_EMPTY
    normalized_session_id = str(cache_session_id or "")
    if not normalized_session_id and normalized_status == PROMPT_CACHE_STATE_READY:
        normalized_status = PROMPT_CACHE_STATE_EMPTY
    return {
        "cache_session_id": normalized_session_id,
        "status": normalized_status,
    }


def create_empty_prompt_cache_state() -> dict[str, Any]:
    return create_prompt_cache_state()


def normalize_prompt_cache_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return create_empty_prompt_cache_state()
    return create_prompt_cache_state(
        cache_session_id=str(state.get("cache_session_id") or ""),
        status=str(state.get("status") or PROMPT_CACHE_STATE_EMPTY),
    )


def get_prompt_cache_session_id(state: Any) -> str:
    return normalize_prompt_cache_state(state).get("cache_session_id", "")


@dataclass(slots=True)
class PromptCacheEntry:
    model_signature: str
    backend: str
    token_ids: list[int]
    cache_length: int
    media_signature: str
    media_sequence: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    prompt_cache: Any = None


@dataclass(slots=True)
class PromptCachePlan:
    cache_session_id: str
    backend: str
    prompt: str
    prompt_token_ids: list[int]
    prompt_cache: Any
    cached_prefix_len: int
    cached_token_prefix_len: int
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


class PromptCachePlanner:
    def prepare_chat_generation(
        self,
        model: Any,
        prompt: str,
        turn_media: list[dict[str, list[str]]],
        cache_session_id: str,
        entry: Optional[PromptCacheEntry],
    ) -> Optional[PromptCachePlan]:
        if self._is_multimodal_model(model):
            return self._prepare_multimodal_generation(model, prompt, turn_media, cache_session_id, entry)
        if self._is_text_model(model):
            return self._prepare_text_generation(model, prompt, cache_session_id, entry)
        return None

    def build_cache_entry(
        self,
        model: Any,
        plan: Optional[PromptCachePlan],
        generated_token_ids: list[int],
    ) -> Optional[PromptCacheEntry]:
        if plan is None:
            return None

        return PromptCacheEntry(
            model_signature=self._get_model_signature(model),
            backend=plan.backend,
            token_ids=plan.prompt_token_ids + list(generated_token_ids),
            cache_length=self._get_prompt_cache_length(plan.prompt_cache),
            media_signature=plan.media_signature,
            media_sequence=list(plan.media_sequence),
            prompt_cache=plan.prompt_cache,
        )

    def _prepare_text_generation(
        self,
        model: Any,
        prompt: str,
        cache_session_id: str,
        entry: Optional[PromptCacheEntry],
    ) -> PromptCachePlan:
        prompt_token_ids = list(model.encode_prompt_tokens(prompt))
        prompt_cache = model.make_prompt_cache()
        cached_prefix_len = 0
        cached_token_prefix_len = 0

        if self._can_reuse_text_cache(model, entry, prompt_token_ids):
            cached = self._copy_prompt_cache(entry.prompt_cache)
            if cached is not None:
                prompt_cache = cached
                cached_token_prefix_len = len(entry.token_ids)
                cached_prefix_len = self._get_reusable_cache_length(entry)

        return PromptCachePlan(
            cache_session_id=cache_session_id,
            backend="text",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_cache=prompt_cache,
            cached_prefix_len=cached_prefix_len,
            cached_token_prefix_len=cached_token_prefix_len,
            media_signature="",
            media_sequence=[],
        )

    def _prepare_multimodal_generation(
        self,
        model: Any,
        prompt: str,
        turn_media: list[dict[str, list[str]]],
        cache_session_id: str,
        entry: Optional[PromptCacheEntry],
    ) -> PromptCachePlan:
        images, audios = self._flatten_turn_media(turn_media)
        prepared_inputs = model.prepare_prompt_inputs(prompt, images=images, audios=audios)
        prompt_token_ids = list(model.extract_prompt_token_ids(prepared_inputs))
        media_sequence = self._normalize_turn_media(turn_media)
        media_signature = self._compute_media_signature(turn_media)
        prompt_cache = model.make_prompt_cache()
        cached_prefix_len = 0
        cached_token_prefix_len = 0

        if self._can_reuse_multimodal_cache(model, entry, prompt_token_ids, media_sequence):
            cached = self._copy_prompt_cache(entry.prompt_cache)
            if cached is not None:
                prompt_cache = cached
                cached_token_prefix_len = len(entry.token_ids)
                cached_prefix_len = self._get_reusable_cache_length(entry)

        return PromptCachePlan(
            cache_session_id=cache_session_id,
            backend="multimodal",
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_cache=prompt_cache,
            cached_prefix_len=cached_prefix_len,
            cached_token_prefix_len=cached_token_prefix_len,
            media_signature=media_signature,
            media_sequence=media_sequence,
            prepared_inputs=prepared_inputs,
        )

    @staticmethod
    def _is_text_model(model: Any) -> bool:
        return callable(getattr(model, "encode_prompt_tokens", None))

    @staticmethod
    def _is_multimodal_model(model: Any) -> bool:
        return callable(getattr(model, "prepare_prompt_inputs", None)) and callable(
            getattr(model, "extract_prompt_token_ids", None)
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

    def _can_reuse_text_cache(
        self,
        model: Any,
        entry: Optional[PromptCacheEntry],
        prompt_token_ids: list[int],
    ) -> bool:
        return (
            entry is not None
            and entry.backend == "text"
            and entry.model_signature == self._get_model_signature(model)
            and bool(entry.prompt_cache)
            and self._get_reusable_cache_length(entry) > 0
            and self._is_prefix(entry.token_ids, prompt_token_ids)
        )

    def _can_reuse_multimodal_cache(
        self,
        model: Any,
        entry: Optional[PromptCacheEntry],
        prompt_token_ids: list[int],
        media_sequence: list[tuple[tuple[str, ...], tuple[str, ...]]],
    ) -> bool:
        return (
            entry is not None
            and entry.backend == "multimodal"
            and entry.model_signature == self._get_model_signature(model)
            and bool(entry.prompt_cache)
            and self._get_reusable_cache_length(entry) > 0
            and self._is_prefix(entry.media_sequence, media_sequence)
            and self._is_prefix(entry.token_ids, prompt_token_ids)
        )

    @staticmethod
    def _is_prefix(prefix: list[Any], values: list[Any]) -> bool:
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
    def _get_model_signature(model: Any) -> str:
        return f"{type(model).__name__}:{getattr(model, 'model_path', '')}"

    @staticmethod
    def _compute_media_signature(turn_media: list[dict[str, list[str]]]) -> str:
        parts: list[str] = []
        for item in turn_media:
            images = [str(path) for path in item.get("images", [])]
            audios = [str(path) for path in item.get("audios", [])]
            parts.append(f"images={'|'.join(images)}\naudios={'|'.join(audios)}")
        return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()

    def _get_reusable_cache_length(self, entry: PromptCacheEntry) -> int:
        if entry.cache_length > 0:
            return max(0, int(entry.cache_length))
        return self._get_prompt_cache_length(entry.prompt_cache)

    def _get_prompt_cache_length(self, prompt_cache: Any) -> int:
        for cache_entry in self._iter_prompt_cache_entries(prompt_cache):
            cache_length = self._get_cache_entry_length(cache_entry)
            if cache_length > 0:
                return cache_length
        return 0

    def _iter_prompt_cache_entries(self, prompt_cache: Any):
        if prompt_cache is None:
            return
        if isinstance(prompt_cache, (list, tuple)):
            for item in prompt_cache:
                yield from self._iter_prompt_cache_entries(item)
            return
        nested_caches = getattr(prompt_cache, "caches", None)
        if isinstance(nested_caches, list):
            for item in nested_caches:
                yield from self._iter_prompt_cache_entries(item)
            return
        yield prompt_cache

    @staticmethod
    def _get_cache_entry_length(cache_entry: Any) -> int:
        if cache_entry is None:
            return 0

        size_fn = getattr(cache_entry, "size", None)
        if callable(size_fn):
            try:
                size_value = size_fn()
            except Exception:
                size_value = 0
            if isinstance(size_value, int):
                return max(0, size_value)
            if hasattr(size_value, "item"):
                try:
                    return max(0, int(size_value.item()))
                except Exception:
                    pass

        offset = getattr(cache_entry, "offset", None)
        if isinstance(offset, int):
            return max(0, offset)
        if hasattr(offset, "item"):
            try:
                return max(0, int(offset.item()))
            except Exception:
                return 0
        return 0


class PromptCacheRegistry:
    def __init__(self, planner: Optional[PromptCachePlanner] = None) -> None:
        self._planner = planner or PromptCachePlanner()
        self._entries: dict[str, PromptCacheEntry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def normalize_state(state: Any) -> dict[str, Any]:
        return normalize_prompt_cache_state(state)

    def prepare_chat_generation(
        self,
        model: Any,
        prompt: str,
        turn_media: list[dict[str, list[str]]],
        state: Any,
    ) -> Optional[PromptCachePlan]:
        normalized_state = normalize_prompt_cache_state(state)
        cache_session_id = normalized_state["cache_session_id"] or uuid.uuid4().hex
        with self._lock:
            entry = self._entries.get(cache_session_id)
        return self._planner.prepare_chat_generation(
            model=model,
            prompt=prompt,
            turn_media=turn_media,
            cache_session_id=cache_session_id,
            entry=entry,
        )

    def commit_chat_generation(
        self,
        model: Any,
        plan: Optional[PromptCachePlan],
        generated_token_ids: list[int],
    ) -> dict[str, Any]:
        if plan is None:
            return create_empty_prompt_cache_state()

        entry = self._planner.build_cache_entry(model=model, plan=plan, generated_token_ids=generated_token_ids)
        if entry is None:
            self.release_chat_cache_session(plan.cache_session_id)
            return create_empty_prompt_cache_state()

        with self._lock:
            self._entries[plan.cache_session_id] = entry
        return create_prompt_cache_state(
            cache_session_id=plan.cache_session_id,
            status=PROMPT_CACHE_STATE_READY,
        )

    def release_chat_cache_session(self, state_or_session_id: Any) -> None:
        cache_session_id = self._resolve_session_id(state_or_session_id)
        if not cache_session_id:
            return

        with self._lock:
            entry = self._entries.pop(cache_session_id, None)

        self._dispose_entries([entry])

    def clear_chat_caches(self) -> None:
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()

        self._dispose_entries(entries)

    def _resolve_session_id(self, state_or_session_id: Any) -> str:
        if isinstance(state_or_session_id, str):
            return state_or_session_id.strip()
        return get_prompt_cache_session_id(state_or_session_id)

    @staticmethod
    def _dispose_entries(entries: list[Optional[PromptCacheEntry]]) -> None:
        released_any = False
        for entry in entries:
            if entry is None:
                continue
            entry.prompt_cache = None
            entry.token_ids = []
            entry.media_sequence = []
            released_any = True

        if not released_any:
            return

        gc.collect()
        mlx.core.clear_cache()
