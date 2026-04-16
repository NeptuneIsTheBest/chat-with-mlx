from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

from ..model import BaseLocalModel, Message, MessageRole
from .chat_types import AttachmentRef, NormalizedTurn, TurnMedia
from .files import FileService
from .structured_output import is_structured_ui_message


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}


class ChatPreprocessor:
    def __init__(self, file_service: FileService) -> None:
        self.file_service = file_service

    def normalize_incoming_message(self, message: Any) -> NormalizedTurn:
        if isinstance(message, dict):
            text = message.get("text", "")
            files = message.get("files", [])
        elif isinstance(message, str):
            text = message
            files = []
        elif message is None:
            text = ""
            files = []
        else:
            text = str(message)
            files = []

        return NormalizedTurn(
            role=MessageRole.USER.value,
            text=self._normalize_text_content(text),
            attachments=self._classify_file_entries(self._normalize_file_entries(files)),
        )

    def prepare_model_inputs(
        self,
        current_message: NormalizedTurn,
        history: list[dict[str, Any]],
        system_prompt: Optional[str],
        model_instance: BaseLocalModel,
    ) -> tuple[str, list[dict[str, Any]], list[TurnMedia]]:
        effective_history: list[dict[str, Any]] = []
        if system_prompt and system_prompt.strip():
            effective_history.append(Message(MessageRole.SYSTEM, content=system_prompt).to_dict())
        effective_history.extend(history)
        effective_history = [
            history_item
            for history_item in effective_history
            if not is_structured_ui_message(history_item)
        ]

        processed_message_text, current_turn_media = self._render_turn(current_message)
        preprocessed_history: list[dict[str, Any]] = []
        history_turn_media: list[TurnMedia] = []
        normalized_history = [self._normalize_history_item(history_item) for history_item in effective_history]

        index = 0
        while index < len(effective_history):
            original_item = effective_history[index]
            normalized_turn = normalized_history[index]
            processed_item = dict(original_item)
            rendered_text, turn_media = self._render_turn(normalized_turn)

            if normalized_turn.attachments and not normalized_turn.text:
                next_text = ""
                consumed_next_item = False
                if index + 1 < len(normalized_history):
                    next_turn = normalized_history[index + 1]
                    next_item = effective_history[index + 1]
                    if next_turn.text and not next_turn.attachments and next_item.get("role") == original_item.get("role"):
                        next_text = next_turn.text
                        consumed_next_item = True

                processed_item["content"] = rendered_text + next_text
                preprocessed_history.append(processed_item)
                history_turn_media.append(turn_media)
                index += 2 if consumed_next_item else 1
                continue

            processed_item["content"] = rendered_text
            preprocessed_history.append(processed_item)
            history_turn_media.append(turn_media)
            index += 1

        turn_media_sequence = history_turn_media + [current_turn_media]
        self.validate_multimodal_inputs(model_instance, turn_media_sequence)
        return processed_message_text, preprocessed_history, turn_media_sequence

    def validate_multimodal_inputs(
        self,
        model_instance: BaseLocalModel,
        turn_media_sequence: list[TurnMedia],
    ) -> None:
        if any(len(turn_media.audios) > 1 for turn_media in turn_media_sequence):
            raise RuntimeError("Only one audio file is supported per message.")

        flat_images = [path for turn_media in turn_media_sequence for path in turn_media.images]
        flat_audios = [path for turn_media in turn_media_sequence for path in turn_media.audios]
        if flat_images and not model_instance.supports_multimodal_ability("vision"):
            raise RuntimeError("The loaded model does not support image inputs.")

        if flat_audios and not model_instance.supports_multimodal_ability("audio"):
            raise RuntimeError("The loaded model does not support audio inputs.")

    def _normalize_history_item(self, history_item: dict[str, Any]) -> NormalizedTurn:
        text_parts: list[str] = []
        file_entries: list[Any] = []
        self._collect_content_parts(history_item.get("content"), text_parts, file_entries)

        files = history_item.get("files", [])
        file_entries.extend(self._normalize_file_entries(files))
        return NormalizedTurn(
            role=str(history_item.get("role") or MessageRole.USER.value),
            text="".join(text_parts),
            attachments=self._classify_file_entries(file_entries),
        )

    def _render_turn(self, turn: NormalizedTurn) -> tuple[str, TurnMedia]:
        document_parts = []
        images: list[str] = []
        audios: list[str] = []

        for attachment in turn.attachments:
            if attachment.kind == "document":
                document_path = Path(attachment.path).expanduser().resolve()
                document_text = self.file_service.read_document_text(document_path)
                if document_text:
                    document_parts.append(self.file_service.render_document_for_prompt(document_path, document_text))
            elif attachment.kind == "image":
                images.append(attachment.path)
            elif attachment.kind == "audio":
                audios.append(attachment.path)

        document_parts.append(turn.text)
        return "".join(document_parts), TurnMedia(images=images, audios=audios)

    @staticmethod
    def _normalize_text_content(text_content: Any) -> str:
        if isinstance(text_content, str):
            return text_content
        return str(text_content) if text_content is not None else ""

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

    @staticmethod
    def _normalize_file_entries(file_entries: Any) -> list[Any]:
        if isinstance(file_entries, list):
            return file_entries
        if isinstance(file_entries, Sequence) and not isinstance(file_entries, (str, bytes, bytearray)):
            return list(file_entries)
        if file_entries:
            return [file_entries]
        return []

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

    def _classify_file_entries(self, file_entries: list[Any]) -> list[AttachmentRef]:
        attachments: list[AttachmentRef] = []
        seen_paths: set[str] = set()

        for file_entry in file_entries:
            normalized_path = self._normalize_file_path(file_entry)
            if not normalized_path:
                continue

            canonical_path = str(Path(normalized_path).expanduser().resolve())
            if canonical_path in seen_paths:
                continue

            seen_paths.add(canonical_path)
            suffix = Path(canonical_path).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                kind = "image"
            elif suffix in AUDIO_EXTENSIONS:
                kind = "audio"
            else:
                kind = "document"
            attachments.append(AttachmentRef(kind=kind, path=canonical_path))
        return attachments
