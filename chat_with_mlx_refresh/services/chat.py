from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional

from ..model import BaseLocalModel, Message, MessageRole, ModelManager, MultimodalModel
from .async_stream import ThreadedGeneratorBridge
from .context_management import ContextManagementService
from .error_handling import raise_gradio_error
from .files import FileService
from .prompt_cache import PromptCacheService, PromptTokenRecorder, create_empty_prompt_cache_state
from .rag import RAGService
from .structured_output import is_structured_ui_message
from .streaming import StreamSession


logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


def ensure_model_has_chat_template(model: BaseLocalModel) -> None:
    if isinstance(model, MultimodalModel):
        return

    tokenizer_to_check = None
    if hasattr(model, "tokenizer"):
        tokenizer_to_check = model.tokenizer

    if tokenizer_to_check and getattr(tokenizer_to_check, "chat_template", None) is None:
        model_name = getattr(model, "model_name", type(model).__name__)
        raise RuntimeError(
            f"Model {model_name} does not have a chat template. "
            "Please use the 'Completion' tab or set a chat template."
        )


def normalize_sampling_params(
    temperature: float,
    top_k: int,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    presence_penalty: float,
) -> dict[str, float]:
    return {
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "min_p": float(min_p),
        "repetition_penalty": float(repetition_penalty),
        "presence_penalty": float(presence_penalty),
    }


def get_eos_token_from_model(model: BaseLocalModel) -> Optional[str]:
    if isinstance(model, MultimodalModel):
        if getattr(model, "processor", None) and getattr(model.processor, "tokenizer", None):
            return model.processor.tokenizer.eos_token
        return None
    return model.tokenizer.eos_token if getattr(model, "tokenizer", None) else None


class ChatService:
    def __init__(
        self,
        model_manager: ModelManager,
        file_service: FileService,
        rag_service: RAGService,
        generation_stop_event: Any,
    ) -> None:
        self.model_manager = model_manager
        self.file_service = file_service
        self.rag_service = rag_service
        self.generation_stop_event = generation_stop_event
        self.context_management = ContextManagementService()
        self.prompt_cache_service = PromptCacheService()

    def get_loaded_model(self) -> BaseLocalModel:
        model = self.model_manager.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    @staticmethod
    def _normalize_incoming_message(message: Any) -> dict[str, Any]:
        if isinstance(message, dict):
            normalized_message = dict(message)
            text = normalized_message.get("text", "")
            files = normalized_message.get("files", [])
        elif isinstance(message, str):
            normalized_message = {}
            text = message
            files = []
        elif message is None:
            normalized_message = {}
            text = ""
            files = []
        else:
            normalized_message = {}
            text = str(message)
            files = []

        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        if isinstance(files, list):
            normalized_files = files
        elif isinstance(files, Sequence) and not isinstance(files, (str, bytes, bytearray)):
            normalized_files = list(files)
        elif files:
            normalized_files = [files]
        else:
            normalized_files = []

        normalized_message["text"] = text
        normalized_message["files"] = normalized_files
        return normalized_message

    @staticmethod
    def _normalize_file_path(file_entry: Any) -> Optional[str]:
        if isinstance(file_entry, str):
            return file_entry
        if isinstance(file_entry, dict):
            path = file_entry.get("path")
            if path:
                return str(path)
            file_value = file_entry.get("file")
            if isinstance(file_value, dict):
                nested_path = file_value.get("path")
                return str(nested_path) if nested_path else None
            if file_value:
                return str(file_value)
            value = file_entry.get("value")
            if isinstance(value, dict):
                nested_path = value.get("path")
                return str(nested_path) if nested_path else None
            if isinstance(value, str) and Path(value).suffix:
                return value
            return None
        path = getattr(file_entry, "path", None)
        if path:
            return str(path)
        return None

    def _collect_content_parts(
        self,
        content: Any,
        text_parts: list[str],
        file_entries: list[Any],
    ) -> None:
        if content is None:
            return

        if isinstance(content, str):
            text_parts.append(content)
            return

        if isinstance(content, dict):
            normalized_path = self._normalize_file_path(content)
            if normalized_path:
                file_entries.append({"path": normalized_path})

            for key in ("text", "value", "content"):
                nested = content.get(key)
                if nested is not None and nested is not content:
                    self._collect_content_parts(nested, text_parts, file_entries)
            return

        if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
            for item in content:
                self._collect_content_parts(item, text_parts, file_entries)
            return

        normalized_path = self._normalize_file_path(content)
        if normalized_path:
            file_entries.append(content)

    def _extract_history_item_content(self, history_item: dict[str, Any]) -> tuple[str, list[Any]]:
        text_parts: list[str] = []
        file_entries: list[Any] = []
        self._collect_content_parts(history_item.get("content"), text_parts, file_entries)

        files = history_item.get("files", [])
        if isinstance(files, list):
            file_entries.extend(files)

        return "".join(text_parts), file_entries

    def _classify_file_entries(self, file_entries: list[Any]) -> tuple[list[str], list[str], list[str]]:
        document_paths: list[str] = []
        image_paths: list[str] = []
        audio_paths: list[str] = []

        for file_entry in file_entries:
            normalized_path = self._normalize_file_path(file_entry)
            if not normalized_path:
                continue

            suffix = Path(normalized_path).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_paths.append(normalized_path)
            elif suffix in AUDIO_EXTENSIONS:
                audio_paths.append(normalized_path)
            else:
                document_paths.append(normalized_path)

        return (
            self._deduplicate_paths(document_paths),
            self._deduplicate_paths(image_paths),
            self._deduplicate_paths(audio_paths),
        )

    def _classify_message_files(self, message: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        return self._classify_file_entries(list(message.get("files", [])))

    def _classify_history_item_files(self, history_item: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        _, file_entries = self._extract_history_item_content(history_item)
        return self._classify_file_entries(file_entries)

    @staticmethod
    def _normalize_text_content(text_content: Any) -> str:
        if isinstance(text_content, str):
            return text_content
        return str(text_content) if text_content is not None else ""

    def _load_document_contents(self, document_paths: list[str]) -> str:
        content_parts = []
        for file_path_str in document_paths:
            if not file_path_str:
                continue
            file_content = self.file_service.load_file(Path(file_path_str))
            if file_content:
                content_parts.append(file_content)
        return "".join(content_parts)

    def _history_item_has_files(self, history_item: dict[str, Any]) -> bool:
        _, file_entries = self._extract_history_item_content(history_item)
        return bool(file_entries)

    def _preprocess_conversation(
        self,
        message: dict[str, Any],
        history: list[dict[str, Any]],
        document_paths: Optional[list[str]] = None,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, list[str]]]]:
        processed_message_text = self._load_document_contents(document_paths or [])
        processed_message_text += self._normalize_text_content(message.get("text", ""))

        preprocessed_history: list[dict[str, Any]] = []
        turn_media: list[dict[str, list[str]]] = []
        index = 0
        while index < len(history):
            current_item_original = history[index]
            current_processed_item = current_item_original.copy()
            current_text_content, current_file_entries = self._extract_history_item_content(current_item_original)
            document_paths_in_item, image_paths_in_item, audio_paths_in_item = self._classify_history_item_files(
                current_item_original
            )
            current_turn_media = {
                "images": image_paths_in_item,
                "audios": audio_paths_in_item,
            }

            if current_file_entries and not current_text_content:
                final_combined_content = self._load_document_contents(document_paths_in_item)
                next_text = ""
                consumed_next_item = False
                if index + 1 < len(history):
                    next_original_item = history[index + 1]
                    next_content, next_file_entries = self._extract_history_item_content(next_original_item)
                    if (
                        next_content
                        and not next_file_entries
                        and next_original_item.get("role") == current_item_original.get("role")
                    ):
                        next_text = next_content
                        consumed_next_item = True

                current_processed_item["content"] = final_combined_content + next_text
                preprocessed_history.append(current_processed_item)
                turn_media.append(current_turn_media)
                index += 2 if consumed_next_item else 1
                continue

            normalized_content = self._normalize_text_content(current_text_content)
            if document_paths_in_item:
                normalized_content = self._load_document_contents(document_paths_in_item) + normalized_content
            current_processed_item["content"] = normalized_content
            preprocessed_history.append(current_processed_item)
            turn_media.append(current_turn_media)
            index += 1

        return processed_message_text, preprocessed_history, turn_media

    @staticmethod
    def _deduplicate_paths(paths: list[str]) -> list[str]:
        seen = set()
        deduplicated = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduplicated.append(path)
        return deduplicated

    def _validate_multimodal_inputs(
        self,
        model_instance: BaseLocalModel,
        turn_media: list[dict[str, list[str]]],
    ) -> None:
        if any(len(item.get("audios", [])) > 1 for item in turn_media):
            raise RuntimeError("Only one audio file is supported per message.")

        image_paths = [path for item in turn_media for path in item.get("images", [])]
        audio_paths = [path for item in turn_media for path in item.get("audios", [])]
        if image_paths and not model_instance.supports_multimodal_ability("vision"):
            raise RuntimeError("The loaded model does not support image inputs.")

        if audio_paths and not model_instance.supports_multimodal_ability("audio"):
            raise RuntimeError("The loaded model does not support audio inputs.")

    @staticmethod
    def _flatten_turn_media(turn_media: list[dict[str, list[str]]]) -> tuple[list[str], list[str]]:
        image_paths = [path for item in turn_media for path in item.get("images", [])]
        audio_paths = [path for item in turn_media for path in item.get("audios", [])]
        return image_paths, audio_paths

    def _prepare_model_inputs(
        self,
        current_message_dict: dict[str, Any],
        history_list: list[dict[str, Any]],
        system_prompt: Optional[str],
        model_instance: BaseLocalModel,
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, list[str]]]]:
        effective_history: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            effective_history.append(Message(MessageRole.SYSTEM, content=system_prompt).to_dict())
        effective_history.extend(history_list)
        effective_history = [
            history_item
            for history_item in effective_history
            if not is_structured_ui_message(history_item)
        ]

        document_paths, image_paths, audio_paths = self._classify_message_files(current_message_dict)
        processed_message_text, processed_history_list, history_turn_media = self._preprocess_conversation(
            current_message_dict,
            effective_history,
            document_paths=document_paths,
        )
        current_turn_media = {"images": image_paths, "audios": audio_paths}
        turn_media = history_turn_media + [current_turn_media]
        self._validate_multimodal_inputs(model_instance, turn_media)
        return processed_message_text, processed_history_list, turn_media

    def _apply_rag_if_needed(self, message: dict[str, Any], rag_enabled: bool, rag_n_results: int) -> dict[str, Any]:
        if not rag_enabled or not message.get("text"):
            return message
        new_message = dict(message)
        new_message["text"] = self.rag_service.enhance_message(message.get("text", ""), rag_enabled, rag_n_results)
        return new_message

    def reset_context_state(self, auto_manage_context: bool = True) -> tuple[dict[str, Any], str]:
        return self.context_management.reset_context_state(auto_manage_context)

    def reset_chat_state(self, auto_manage_context: bool = True) -> tuple[dict[str, Any], str, dict[str, Any]]:
        summary_state, status_text = self.reset_context_state(auto_manage_context)
        return summary_state, status_text, create_empty_prompt_cache_state()

    def handle_chat(
        self,
        message: Any,
        history: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        rag_enabled: bool = False,
        rag_n_results: int = 5,
        auto_manage_context: bool = True,
        context_summary_state: Optional[dict[str, Any]] = None,
        prompt_cache_state: Optional[dict[str, Any]] = None,
        stream: bool = True,
    ) -> Iterator[Any]:
        try:
            message = self._normalize_incoming_message(message)
            message = self._apply_rag_if_needed(message, rag_enabled, rag_n_results)
            with self.model_manager.reserve_loaded_model() as model:
                ensure_model_has_chat_template(model)

                processed_message_text, processed_history_list, turn_media = self._prepare_model_inputs(
                    message,
                    history,
                    system_prompt,
                    model,
                )
                context_result = self.context_management.manage_chat_context(
                    model=model,
                    message_text=processed_message_text,
                    history=processed_history_list,
                    turn_media=turn_media,
                    max_tokens=max_tokens,
                    auto_manage_context=auto_manage_context,
                    summary_state=context_summary_state,
                )
                image_paths, audio_paths = self._flatten_turn_media(context_result.turn_media)
                sampling = normalize_sampling_params(
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    presence_penalty,
                )
                response_args = {
                    "message": processed_message_text,
                    "history": context_result.history,
                    "stream": stream,
                    "max_tokens": max_tokens,
                    **sampling,
                }
                current_message = Message(MessageRole.USER, processed_message_text).to_dict()
                formatted_prompt = model.format_chat_prompt(
                    message=current_message,
                    history=context_result.history,
                    turn_media=context_result.turn_media,
                )
                response_args["formatted_prompt"] = formatted_prompt

                cache_plan = self.prompt_cache_service.prepare_chat_generation(
                    model=model,
                    prompt=formatted_prompt,
                    turn_media=context_result.turn_media,
                    state=prompt_cache_state,
                )
                if cache_plan is not None:
                    response_args["prompt_token_ids"] = cache_plan.prompt_token_ids
                    response_args["prompt_cache"] = cache_plan.prompt_cache
                    response_args["cached_prefix_len"] = cache_plan.cached_prefix_len
                    response_args["cached_token_prefix_len"] = cache_plan.cached_token_prefix_len
                    if cache_plan.prepared_inputs is not None:
                        response_args["prepared_inputs"] = cache_plan.prepared_inputs

                if isinstance(model, MultimodalModel):
                    response_args["images"] = image_paths
                    response_args["audios"] = audio_paths
                    response_args["turn_media"] = context_result.turn_media

                prompt_token_recorder = PromptTokenRecorder()
                session = StreamSession(
                    response_stream=model.generate_response(**response_args),
                    eos_token=get_eos_token_from_model(model),
                    stream=stream,
                    generation_stop_event=self.generation_stop_event,
                    thinking_title="Thinking",
                    inline_thought_title="Thinking",
                    thinking_id=0,
                    base_messages=[],
                    chunk_observer=prompt_token_recorder.observe,
                )
                transient_prompt_cache_state = create_empty_prompt_cache_state()
                last_payload = None
                for payload in session:
                    last_payload = payload
                    yield payload, context_result.summary_state, context_result.status_text, transient_prompt_cache_state

                final_payload = session.final_messages or last_payload
                if final_payload is not None and final_payload != last_payload:
                    yield final_payload, context_result.summary_state, context_result.status_text, transient_prompt_cache_state

                if cache_plan is not None and not self.generation_stop_event.is_set():
                    updated_prompt_cache_state = self.prompt_cache_service.build_success_state(
                        model=model,
                        plan=cache_plan,
                        generated_token_ids=prompt_token_recorder.token_ids,
                    )
                    if final_payload is not None:
                        yield final_payload, context_result.summary_state, context_result.status_text, updated_prompt_cache_state
        except Exception as exc:
            raise_gradio_error(exc, logger=logger, action="handle chat requests")

    async def managed_chat_generator(
        self,
        message: Any,
        history: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        top_k: int = 20,
        top_p: float = 0.95,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        rag_enabled: bool = False,
        rag_n_results: int = 5,
        auto_manage_context: bool = True,
        context_summary_state: Optional[dict[str, Any]] = None,
        prompt_cache_state: Optional[dict[str, Any]] = None,
        stream: bool = True,
    ) -> AsyncIterator[Any]:
        try:
            generator = self.handle_chat(
                message=message,
                history=history,
                system_prompt=system_prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                rag_enabled=rag_enabled,
                rag_n_results=rag_n_results,
                auto_manage_context=auto_manage_context,
                context_summary_state=context_summary_state,
                prompt_cache_state=prompt_cache_state,
                stream=stream,
            )
            await asyncio.to_thread(self.model_manager.close_active_generator)
            self.generation_stop_event.clear()
            bridge = ThreadedGeneratorBridge(generator, self.generation_stop_event)
            self.model_manager.set_active_generator(bridge)
            try:
                async for chunk in bridge:
                    yield chunk
            finally:
                bridge.close(wait=False)
                await asyncio.to_thread(bridge.wait_closed)
                self.model_manager.remove_active_generator(bridge)
        except Exception as exc:
            raise_gradio_error(exc, logger=logger, action="stream chat responses")
