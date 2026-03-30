from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from ..model import BaseLocalModel, Message, MessageRole, ModelManager, MultimodalModel
from .async_stream import ThreadedGeneratorBridge
from .error_handling import raise_gradio_error
from .files import FileService
from .rag import RAGService
from .streaming import generate_response_and_stream


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
) -> Dict[str, float]:
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

    def get_loaded_model(self) -> BaseLocalModel:
        model = self.model_manager.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    @staticmethod
    def _normalize_file_path(file_entry: Any) -> Optional[str]:
        if isinstance(file_entry, str):
            return file_entry
        if isinstance(file_entry, dict):
            path = file_entry.get("path")
            return str(path) if path else None
        path = getattr(file_entry, "path", None)
        if path:
            return str(path)
        return None

    def _classify_file_entries(self, file_entries: List[Any]) -> Tuple[List[str], List[str], List[str]]:
        document_paths: List[str] = []
        image_paths: List[str] = []
        audio_paths: List[str] = []

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

    def _classify_message_files(self, message: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        return self._classify_file_entries(list(message.get("files", [])))

    def _classify_history_item_files(self, history_item: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        file_entries: List[Any] = []
        current_content = history_item.get("content")
        if isinstance(current_content, tuple):
            file_entries.extend(list(current_content))

        files = history_item.get("files", [])
        if isinstance(files, list):
            file_entries.extend(files)

        return self._classify_file_entries(file_entries)

    @staticmethod
    def _normalize_text_content(text_content: Any) -> str:
        if isinstance(text_content, str):
            return text_content
        return str(text_content) if text_content is not None else ""

    def _load_document_contents(self, document_paths: List[str]) -> str:
        content_parts = []
        for file_path_str in document_paths:
            if not file_path_str:
                continue
            file_content = self.file_service.load_file(Path(file_path_str))
            if file_content:
                content_parts.append(file_content)
        return "".join(content_parts)

    def _history_item_has_files(self, history_item: Dict[str, Any]) -> bool:
        current_content = history_item.get("content")
        if isinstance(current_content, tuple) and current_content:
            return True

        files = history_item.get("files", [])
        return isinstance(files, list) and bool(files)

    def _preprocess_conversation(
        self,
        message: Dict[str, Any],
        history: List[Dict[str, Any]],
        document_paths: Optional[List[str]] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, List[str]]]]:
        processed_message_text = self._load_document_contents(document_paths or [])
        processed_message_text += self._normalize_text_content(message.get("text", ""))

        preprocessed_history: List[Dict[str, Any]] = []
        turn_media: List[Dict[str, List[str]]] = []
        index = 0
        while index < len(history):
            current_item_original = history[index]
            current_processed_item = current_item_original.copy()
            current_content = current_item_original.get("content")
            document_paths_in_item, image_paths_in_item, audio_paths_in_item = self._classify_history_item_files(
                current_item_original
            )
            current_turn_media = {
                "images": image_paths_in_item,
                "audios": audio_paths_in_item,
            }

            if isinstance(current_content, tuple):
                final_combined_content = self._load_document_contents(document_paths_in_item)
                next_text = ""
                consumed_next_item = False
                if index + 1 < len(history):
                    next_original_item = history[index + 1]
                    next_content = next_original_item.get("content")
                    if (
                        isinstance(next_content, str)
                        and not self._history_item_has_files(next_original_item)
                        and next_original_item.get("role") == current_item_original.get("role")
                    ):
                        next_text = next_content
                        consumed_next_item = True

                current_processed_item["content"] = final_combined_content + next_text
                preprocessed_history.append(current_processed_item)
                turn_media.append(current_turn_media)
                index += 2 if consumed_next_item else 1
                continue

            normalized_content = self._normalize_text_content(current_content)
            if document_paths_in_item:
                normalized_content = self._load_document_contents(document_paths_in_item) + normalized_content
            current_processed_item["content"] = normalized_content
            preprocessed_history.append(current_processed_item)
            turn_media.append(current_turn_media)
            index += 1

        return processed_message_text, preprocessed_history, turn_media

    @staticmethod
    def _deduplicate_paths(paths: List[str]) -> List[str]:
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
        turn_media: List[Dict[str, List[str]]],
    ) -> None:
        if any(len(item.get("audios", [])) > 1 for item in turn_media):
            raise RuntimeError("Only one audio file is supported per message.")

        image_paths = [path for item in turn_media for path in item.get("images", [])]
        audio_paths = [path for item in turn_media for path in item.get("audios", [])]
        if image_paths and not model_instance.supports_multimodal_ability("vision"):
            raise RuntimeError("The loaded model does not support image inputs.")

        if audio_paths and not model_instance.supports_multimodal_ability("audio"):
            raise RuntimeError("The loaded model does not support audio inputs.")

    def _prepare_model_inputs(
        self,
        current_message_dict: Dict[str, Any],
        history_list: List[Dict[str, Any]],
        system_prompt: Optional[str],
        model_instance: BaseLocalModel,
    ) -> Tuple[str, List[Dict[str, Any]], List[str], List[str], List[Dict[str, List[str]]]]:
        effective_history: List[Dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            effective_history.append(Message(MessageRole.SYSTEM, content=system_prompt).to_dict())
        effective_history.extend(history_list)
        effective_history = [
            history_item
            for history_item in effective_history
            if not (
                isinstance(history_item, dict)
                and isinstance(history_item.get("metadata"), dict)
                and history_item["metadata"].get("title") in ["Thinking"]
            )
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

        flattened_image_paths = [path for item in turn_media for path in item.get("images", [])]
        flattened_audio_paths = [path for item in turn_media for path in item.get("audios", [])]
        return processed_message_text, processed_history_list, flattened_image_paths, flattened_audio_paths, turn_media

    def _apply_rag_if_needed(self, message: Dict[str, Any], rag_enabled: bool, rag_n_results: int) -> Dict[str, Any]:
        if not rag_enabled or not message.get("text"):
            return message
        new_message = dict(message)
        new_message["text"] = self.rag_service.enhance_message(message.get("text", ""), rag_enabled, rag_n_results)
        return new_message

    def handle_chat(
        self,
        message: Dict[str, Any],
        history: List[Dict[str, Any]],
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
        stream: bool = True,
    ) -> Iterator[Any]:
        try:
            message = self._apply_rag_if_needed(message, rag_enabled, rag_n_results)
            model = self.get_loaded_model()
            ensure_model_has_chat_template(model)

            processed_message_text, processed_history_list, image_paths, audio_paths, turn_media = self._prepare_model_inputs(
                message,
                history,
                system_prompt,
                model,
            )
            sampling = normalize_sampling_params(temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty)
            response_args = {
                "message": processed_message_text,
                "history": processed_history_list,
                "stream": stream,
                "max_tokens": max_tokens,
                **sampling,
            }
            if isinstance(model, MultimodalModel):
                response_args["images"] = image_paths
                response_args["audios"] = audio_paths
                response_args["turn_media"] = turn_media

            yield from generate_response_and_stream(
                model=model,
                response_args=response_args,
                eos_token=get_eos_token_from_model(model),
                stream=stream,
                generation_stop_event=self.generation_stop_event,
                thinking_title="Thinking",
                inline_thought_title="Thinking",
                thinking_id=0,
                base_messages=[],
            )
        except Exception as exc:
            raise_gradio_error(exc, logger=logger, action="handle chat requests")

    async def managed_chat_generator(
        self,
        message: Dict[str, Any],
        history: List[Dict[str, Any]],
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
                bridge.close()
                self.model_manager.remove_active_generator(bridge)
        except Exception as exc:
            raise_gradio_error(exc, logger=logger, action="stream chat responses")
