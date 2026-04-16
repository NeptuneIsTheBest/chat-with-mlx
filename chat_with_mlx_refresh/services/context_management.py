from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from ..model import BaseLocalModel, Message, MessageRole, MultimodalModel, TextModel
from .chat_prompting import ChatPromptBuilder
from .chat_types import PreparedPrompt, TurnMedia


logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_WINDOW = 32768
PROMPT_TOKEN_RESERVE = 32
RECENT_RAW_MESSAGE_COUNT = 6
SUMMARY_MAX_TOKENS = 256
SUMMARY_LABEL = "Conversation summary (auto-generated, lossy)"


def create_empty_context_summary_state() -> dict[str, Any]:
    return {
        "summary_text": "",
        "covered_message_count": 0,
        "prefix_signature": "",
        "system_prompt_signature": "",
        "model_signature": "",
    }


def get_default_context_status(auto_manage_context: bool = True) -> str:
    if auto_manage_context:
        return "Full history will be used until the context window is approached."
    return "Automatic context management is disabled."


@dataclass(slots=True)
class ContextManagementResult:
    history: list[dict[str, Any]]
    turn_media: list[TurnMedia]
    prepared_prompt: PreparedPrompt
    prompt_tokens: int
    available_prompt_tokens: int
    context_window: int
    summary_state: dict[str, Any]
    status_text: str


class ContextManagementService:
    def __init__(self, prompt_builder: Optional[ChatPromptBuilder] = None) -> None:
        self.prompt_builder = prompt_builder or ChatPromptBuilder()

    def manage_chat_context(
        self,
        model: BaseLocalModel,
        message_text: str,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
        max_tokens: int,
        auto_manage_context: bool,
        summary_state: Optional[dict[str, Any]],
    ) -> ContextManagementResult:
        normalized_state = self._normalize_summary_state(summary_state)
        context_window = self._get_context_window(model)
        available_prompt_tokens = context_window - int(max_tokens) - PROMPT_TOKEN_RESERVE
        if available_prompt_tokens <= 0:
            raise RuntimeError(
                "The selected max_tokens value leaves no room for the prompt. "
                "Lower max_tokens before sending the message."
            )

        current_turn_media = turn_media[-1] if turn_media else TurnMedia()
        history_turn_media = turn_media[:-1] if turn_media else []
        system_message, system_media, non_system_history, non_system_media = self._split_system_message(
            history,
            history_turn_media,
        )
        system_prompt_tokens, _ = self._prepare_prompt(
            model=model,
            message_text=message_text,
            history=self._compose_history(system_message, []),
            turn_media=([system_media] if system_message else []) + [current_turn_media],
        )
        if system_prompt_tokens > available_prompt_tokens:
            raise RuntimeError(
                "The current input plus the active system prompt exceed the available context window. "
                "Shorten the message, reduce attached document content, lower RAG usage, or reduce max_tokens."
            )

        full_prompt_tokens, full_prepared_prompt = self._prepare_prompt(
            model=model,
            message_text=message_text,
            history=history,
            turn_media=turn_media,
        )
        if full_prompt_tokens <= available_prompt_tokens:
            return ContextManagementResult(
                history=history,
                turn_media=turn_media,
                prepared_prompt=full_prepared_prompt,
                prompt_tokens=full_prompt_tokens,
                available_prompt_tokens=available_prompt_tokens,
                context_window=context_window,
                summary_state=normalized_state,
                status_text=(
                    f"Full history fits within the current context window "
                    f"({full_prompt_tokens}/{available_prompt_tokens} prompt tokens)."
                ),
            )

        if not auto_manage_context:
            raise RuntimeError(
                "The conversation exceeds the available context window and automatic context management is disabled. "
                "Enable it in Advanced Setting, shorten the chat history, or lower max_tokens."
            )

        raw_tail_count = min(RECENT_RAW_MESSAGE_COUNT, len(non_system_history))
        old_history = non_system_history[: len(non_system_history) - raw_tail_count]
        old_media = non_system_media[: len(non_system_media) - raw_tail_count]
        raw_tail_history = non_system_history[len(non_system_history) - raw_tail_count :]
        raw_tail_media = non_system_media[len(non_system_media) - raw_tail_count :]

        fallback_status = "Automatic summary could not free enough room, so only the most recent turns were kept"
        try:
            updated_state = self._build_summary_state(
                model=model,
                old_history=old_history,
                old_media=old_media,
                system_message=system_message,
                summary_state=normalized_state,
            )
            managed_history, managed_turn_media = self._compose_managed_history(
                system_message=system_message,
                system_media=system_media,
                summary_text=updated_state["summary_text"],
                raw_history=raw_tail_history,
                raw_media=raw_tail_media,
                current_turn_media=current_turn_media,
            )
            prompt_tokens, prepared_prompt = self._prepare_prompt(
                model=model,
                message_text=message_text,
                history=managed_history,
                turn_media=managed_turn_media,
            )
            if prompt_tokens <= available_prompt_tokens:
                return ContextManagementResult(
                    history=managed_history,
                    turn_media=managed_turn_media,
                    prepared_prompt=prepared_prompt,
                    prompt_tokens=prompt_tokens,
                    available_prompt_tokens=available_prompt_tokens,
                    context_window=context_window,
                    summary_state=updated_state,
                    status_text=(
                        f"Older turns were compressed automatically to fit the context window "
                        f"({prompt_tokens}/{available_prompt_tokens} prompt tokens)."
                    ),
                )

            contracted_history = list(raw_tail_history)
            contracted_media = list(raw_tail_media)
            while contracted_history:
                contracted_history.pop(0)
                contracted_media.pop(0)
                managed_history, managed_turn_media = self._compose_managed_history(
                    system_message=system_message,
                    system_media=system_media,
                    summary_text=updated_state["summary_text"],
                    raw_history=contracted_history,
                    raw_media=contracted_media,
                    current_turn_media=current_turn_media,
                )
                prompt_tokens, prepared_prompt = self._prepare_prompt(
                    model=model,
                    message_text=message_text,
                    history=managed_history,
                    turn_media=managed_turn_media,
                )
                if prompt_tokens <= available_prompt_tokens:
                    return ContextManagementResult(
                        history=managed_history,
                        turn_media=managed_turn_media,
                        prepared_prompt=prepared_prompt,
                        prompt_tokens=prompt_tokens,
                        available_prompt_tokens=available_prompt_tokens,
                        context_window=context_window,
                        summary_state=updated_state,
                        status_text=(
                            f"Older turns were summarized and the raw tail was trimmed to fit "
                            f"({prompt_tokens}/{available_prompt_tokens} prompt tokens)."
                        ),
                    )
        except Exception as exc:
            logger.warning("Automatic conversation summarization failed: %s", exc)
            fallback_status = "Summary generation failed, so only the most recent turns were kept"

        fallback_history = list(raw_tail_history)
        fallback_media = list(raw_tail_media)
        while True:
            managed_history = self._compose_history(system_message, fallback_history)
            managed_turn_media = self._compose_turn_media(system_message, system_media, fallback_media, current_turn_media)
            prompt_tokens, prepared_prompt = self._prepare_prompt(
                model=model,
                message_text=message_text,
                history=managed_history,
                turn_media=managed_turn_media,
            )
            if prompt_tokens <= available_prompt_tokens:
                return ContextManagementResult(
                    history=managed_history,
                    turn_media=managed_turn_media,
                    prepared_prompt=prepared_prompt,
                    prompt_tokens=prompt_tokens,
                    available_prompt_tokens=available_prompt_tokens,
                    context_window=context_window,
                    summary_state=create_empty_context_summary_state(),
                    status_text=(
                        f"{fallback_status} "
                        f"({prompt_tokens}/{available_prompt_tokens} prompt tokens)."
                    ),
                )
            if not fallback_history:
                break
            fallback_history.pop(0)
            fallback_media.pop(0)

        raise RuntimeError(
            "The current input still exceeds the available context window after dropping prior conversation history. "
            "Shorten the message, reduce attached document content, lower RAG usage, or reduce max_tokens."
        )

    def reset_context_state(self, auto_manage_context: bool = True) -> tuple[dict[str, Any], str]:
        return create_empty_context_summary_state(), get_default_context_status(auto_manage_context)

    def _normalize_summary_state(self, state: Optional[dict[str, Any]]) -> dict[str, Any]:
        normalized = create_empty_context_summary_state()
        if not isinstance(state, dict):
            return normalized
        normalized.update(
            {
                "summary_text": str(state.get("summary_text") or ""),
                "covered_message_count": max(0, int(state.get("covered_message_count") or 0)),
                "prefix_signature": str(state.get("prefix_signature") or ""),
                "system_prompt_signature": str(state.get("system_prompt_signature") or ""),
                "model_signature": str(state.get("model_signature") or ""),
            }
        )
        return normalized

    def _get_context_window(self, model: BaseLocalModel) -> int:
        max_length = getattr(model, "max_position_embeddings", None) or DEFAULT_CONTEXT_WINDOW
        return max(int(max_length), 1)

    def _split_system_message(
        self,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> tuple[Optional[dict[str, Any]], TurnMedia, list[dict[str, Any]], list[TurnMedia]]:
        if history and history[0].get("role") == MessageRole.SYSTEM.value:
            system_media = turn_media[0] if turn_media else TurnMedia()
            return history[0], system_media, history[1:], turn_media[1:]
        return None, TurnMedia(), list(history), list(turn_media)

    def _compose_history(
        self,
        system_message: Optional[dict[str, Any]],
        raw_history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        if system_message is not None:
            history.append(system_message)
        history.extend(raw_history)
        return history

    def _compose_turn_media(
        self,
        system_message: Optional[dict[str, Any]],
        system_media: TurnMedia,
        raw_media: list[TurnMedia],
        current_turn_media: TurnMedia,
    ) -> list[TurnMedia]:
        turn_media: list[TurnMedia] = []
        if system_message is not None:
            turn_media.append(system_media)
        turn_media.extend(raw_media)
        turn_media.append(current_turn_media)
        return turn_media

    def _compose_managed_history(
        self,
        system_message: Optional[dict[str, Any]],
        system_media: TurnMedia,
        summary_text: str,
        raw_history: list[dict[str, Any]],
        raw_media: list[TurnMedia],
        current_turn_media: TurnMedia,
    ) -> tuple[list[dict[str, Any]], list[TurnMedia]]:
        managed_history: list[dict[str, Any]] = []
        managed_turn_media: list[TurnMedia] = []

        if system_message is not None:
            merged_system = dict(system_message)
            merged_system["content"] = self._merge_summary_into_system_prompt(system_message.get("content", ""), summary_text)
            managed_history.append(merged_system)
            managed_turn_media.append(system_media)
        elif summary_text:
            managed_history.append(Message(MessageRole.SYSTEM, self._format_summary_block(summary_text)).to_dict())
            managed_turn_media.append(TurnMedia())

        managed_history.extend(raw_history)
        managed_turn_media.extend(raw_media)
        managed_turn_media.append(current_turn_media)
        return managed_history, managed_turn_media

    def _build_summary_state(
        self,
        model: BaseLocalModel,
        old_history: list[dict[str, Any]],
        old_media: list[TurnMedia],
        system_message: Optional[dict[str, Any]],
        summary_state: dict[str, Any],
    ) -> dict[str, Any]:
        if not old_history:
            return create_empty_context_summary_state()

        model_signature = self._get_model_signature(model)
        system_signature = self._get_system_prompt_signature(system_message)
        reusable_summary = ""
        covered_message_count = 0
        existing_count = summary_state.get("covered_message_count", 0)
        if (
            summary_state.get("summary_text")
            and summary_state.get("model_signature") == model_signature
            and summary_state.get("system_prompt_signature") == system_signature
            and 0 < existing_count <= len(old_history)
        ):
            existing_prefix_signature = self._compute_history_signature(old_history[:existing_count], old_media[:existing_count])
            if existing_prefix_signature == summary_state.get("prefix_signature"):
                reusable_summary = str(summary_state.get("summary_text") or "")
                covered_message_count = existing_count

        messages_to_summarize = old_history[covered_message_count:]
        media_to_summarize = old_media[covered_message_count:]
        summary_text = reusable_summary
        if messages_to_summarize:
            summary_text = self._summarize_messages(
                model=model,
                existing_summary=reusable_summary,
                messages=messages_to_summarize,
                turn_media=media_to_summarize,
                context_window=self._get_context_window(model),
            )

        if not summary_text.strip():
            return create_empty_context_summary_state()

        covered_total = len(old_history)
        return {
            "summary_text": summary_text.strip(),
            "covered_message_count": covered_total,
            "prefix_signature": self._compute_history_signature(old_history, old_media),
            "system_prompt_signature": system_signature,
            "model_signature": model_signature,
        }

    def _prepare_prompt(
        self,
        model: BaseLocalModel,
        message_text: str,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> tuple[int, PreparedPrompt]:
        return self.prompt_builder.count_prompt_tokens(
            model=model,
            message_text=message_text,
            history=history,
            turn_media=turn_media,
        )

    def _summarize_messages(
        self,
        model: BaseLocalModel,
        existing_summary: str,
        messages: list[dict[str, Any]],
        turn_media: list[TurnMedia],
        context_window: int,
    ) -> str:
        max_source_tokens = max(1, min(4096, math.floor(context_window * 0.25)))
        rendered_messages = self._render_messages_for_summary(messages, turn_media)
        summary = existing_summary.strip()
        batch_blocks: list[str] = []

        for block in rendered_messages:
            candidate_blocks = batch_blocks + [block]
            source_text = self._build_summary_source(summary, candidate_blocks)
            if self._count_text_tokens(model, source_text) > max_source_tokens and batch_blocks:
                summary = self._run_summary_pass(model, summary, batch_blocks)
                batch_blocks = [self._truncate_text_to_tokens(model, block, max_source_tokens // 2 or 1)]
                continue

            if self._count_text_tokens(model, source_text) > max_source_tokens:
                batch_blocks.append(self._truncate_text_to_tokens(model, block, max_source_tokens // 2 or 1))
            else:
                batch_blocks = candidate_blocks

        if batch_blocks or not summary:
            summary = self._run_summary_pass(model, summary, batch_blocks)

        if not summary.strip():
            raise RuntimeError("The model returned an empty summary.")
        return summary.strip()

    def _run_summary_pass(self, model: BaseLocalModel, existing_summary: str, blocks: list[str]) -> str:
        prompt = self._build_summary_prompt(existing_summary, blocks)
        summary = model.generate_completion(
            prompt=prompt,
            stream=False,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            min_p=0.0,
            max_tokens=SUMMARY_MAX_TOKENS,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )
        if not isinstance(summary, str):
            raise RuntimeError("Unexpected summary output type.")
        cleaned = summary.strip()
        if not cleaned:
            raise RuntimeError("Empty summary output.")
        return cleaned

    def _build_summary_prompt(self, existing_summary: str, blocks: list[str]) -> str:
        prior_summary = existing_summary.strip() or "(none)"
        chunk_text = "\n\n".join(blocks).strip() or "(none)"
        return (
            "Compress the earlier conversation into a short factual memory for continued chat.\n"
            "Keep only durable information: user goals, constraints, decisions, errors, outputs, file names, and "
            "important multimodal references such as image/audio counts.\n"
            "Drop pleasantries, repetition, and reasoning traces.\n"
            "Return plain text bullet lines only.\n\n"
            f"Existing summary:\n{prior_summary}\n\n"
            f"New conversation turns:\n{chunk_text}\n\n"
            "Updated summary:\n"
        )

    def _build_summary_source(self, existing_summary: str, blocks: list[str]) -> str:
        return self._build_summary_prompt(existing_summary, blocks)

    def _render_messages_for_summary(
        self,
        messages: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> list[str]:
        rendered: list[str] = []
        for index, message in enumerate(messages):
            role = str(message.get("role") or MessageRole.USER.value).upper()
            content = str(message.get("content") or "").strip()
            media = turn_media[index] if index < len(turn_media) else TurnMedia()
            media_parts = []
            if media.images:
                media_parts.append(f"images={len(media.images)}")
            if media.audios:
                media_parts.append(f"audios={len(media.audios)}")
            media_suffix = f" [{' '.join(media_parts)}]" if media_parts else ""
            if content:
                rendered.append(f"{role}{media_suffix}\n{content}")
            elif media_suffix:
                rendered.append(f"{role}{media_suffix}")
        return rendered

    def _format_summary_block(self, summary_text: str) -> str:
        return f"{SUMMARY_LABEL}:\n{summary_text.strip()}"

    def _merge_summary_into_system_prompt(self, system_prompt: Any, summary_text: str) -> str:
        cleaned_system_prompt = str(system_prompt or "").strip()
        if not summary_text.strip():
            return cleaned_system_prompt
        summary_block = self._format_summary_block(summary_text)
        if not cleaned_system_prompt:
            return summary_block
        return f"{cleaned_system_prompt}\n\n{summary_block}"

    def _get_model_signature(self, model: BaseLocalModel) -> str:
        return f"{type(model).__name__}:{getattr(model, 'model_path', '')}"

    def _get_system_prompt_signature(self, system_message: Optional[dict[str, Any]]) -> str:
        content = ""
        if system_message is not None:
            content = str(system_message.get("content") or "")
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _compute_history_signature(
        self,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> str:
        parts: list[str] = []
        for index, item in enumerate(history):
            role = str(item.get("role") or "")
            content = str(item.get("content") or "")
            media = turn_media[index] if index < len(turn_media) else TurnMedia()
            parts.append(
                f"{role}\n{content}\nimages={len(media.images)}\naudios={len(media.audios)}"
            )
        return hashlib.sha256("\n---\n".join(parts).encode("utf-8")).hexdigest()

    def _get_text_tokenizer(self, model: BaseLocalModel):
        if isinstance(model, MultimodalModel):
            return model.processor.tokenizer if hasattr(model.processor, "tokenizer") else model.processor
        if isinstance(model, TextModel):
            return model.tokenizer
        return getattr(model, "tokenizer", None) or getattr(model, "processor", None)

    def _count_text_tokens(self, model: BaseLocalModel, text: str) -> int:
        tokenizer = self._get_text_tokenizer(model)
        return len(self._encode_text(tokenizer, text))

    def _truncate_text_to_tokens(self, model: BaseLocalModel, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        tokenizer = self._get_text_tokenizer(model)
        token_ids = self._encode_text(tokenizer, text)
        if len(token_ids) <= max_tokens:
            return text
        truncated = token_ids[:max_tokens]
        try:
            return tokenizer.decode(truncated, skip_special_tokens=True).strip()
        except TypeError:
            return tokenizer.decode(truncated).strip()

    @staticmethod
    def _encode_text(tokenizer: Any, text: str) -> list[int]:
        if tokenizer is None:
            raise RuntimeError("No tokenizer available for token counting.")
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
