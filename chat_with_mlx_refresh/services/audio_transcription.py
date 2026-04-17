from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import BaseLocalModel, MultimodalModel
from .chat_prompting import ChatPromptBuilder
from .chat_types import TurnMedia


AUDIO_TRANSCRIPT_BLOCK_LABEL = "Earlier audio transcript"
TRANSCRIPTION_PROMPT = (
    "Transcribe the attached audio in its original language. "
    "Return only the transcript text. "
    "Do not add summaries, labels, or commentary."
)
TRANSCRIPTION_MAX_TOKENS = 2048


def create_audio_transcript_state(entries: dict[str, str] | None = None) -> dict[str, Any]:
    normalized_entries: dict[str, str] = {}
    if isinstance(entries, dict):
        for key, value in entries.items():
            normalized_key = str(key or "").strip()
            normalized_value = str(value or "").strip()
            if normalized_key and normalized_value:
                normalized_entries[normalized_key] = normalized_value
    return {"entries": normalized_entries}


def create_empty_audio_transcript_state() -> dict[str, Any]:
    return create_audio_transcript_state()


def normalize_audio_transcript_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return create_empty_audio_transcript_state()
    return create_audio_transcript_state(entries=state.get("entries"))


@dataclass(slots=True)
class AudioRetentionResult:
    message_text: str
    history: list[dict[str, Any]]
    turn_media: list[TurnMedia]
    transcript_state: dict[str, Any]
    retained_transcript_turn_count: int


class AudioTranscriber(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError


class CurrentModelAudioTranscriber(AudioTranscriber):
    def __init__(self, model: MultimodalModel, prompt_builder: ChatPromptBuilder) -> None:
        self.model = model
        self.prompt_builder = prompt_builder

    def transcribe(self, audio_path: str) -> str:
        if not self.model.supports_multimodal_ability("audio"):
            raise RuntimeError("The loaded model does not support audio transcription.")

        prepared_prompt = self.prompt_builder.prepare_prompt(
            model=self.model,
            message_text=TRANSCRIPTION_PROMPT,
            history=[],
            turn_media=[TurnMedia(audios=[audio_path])],
        )
        transcript = self.model.perform_generation(
            prepared_prompt.formatted_prompt,
            stream=False,
            prepared_inputs=prepared_prompt.prepared_inputs,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            min_p=0.0,
            max_tokens=TRANSCRIPTION_MAX_TOKENS,
            repetition_penalty=1.0,
            presence_penalty=0.0,
        )
        if not isinstance(transcript, str):
            raise RuntimeError("Unexpected audio transcription output type.")

        cleaned = transcript.strip()
        if not cleaned:
            raise RuntimeError("The model returned an empty transcript for an earlier audio turn.")
        return cleaned


class AudioTranscriptRetainer:
    def __init__(self, prompt_builder: ChatPromptBuilder) -> None:
        self.prompt_builder = prompt_builder

    def retain_audio_context(
        self,
        model: BaseLocalModel,
        message_text: str,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
        transcript_state: Any,
    ) -> AudioRetentionResult:
        normalized_state = normalize_audio_transcript_state(transcript_state)
        prompt_audio_limit = model.get_prompt_audio_limit()
        if prompt_audio_limit is None:
            return AudioRetentionResult(
                message_text=message_text,
                history=history,
                turn_media=turn_media,
                transcript_state=normalized_state,
                retained_transcript_turn_count=0,
            )
        if not isinstance(model, MultimodalModel):
            raise RuntimeError("Audio prompt limits are only supported for multimodal models.")

        normalized_limit = max(0, int(prompt_audio_limit))
        audio_turn_indexes = [index for index, media in enumerate(turn_media) if media.audios]
        if len(audio_turn_indexes) <= normalized_limit:
            return AudioRetentionResult(
                message_text=message_text,
                history=history,
                turn_media=turn_media,
                transcript_state=normalized_state,
                retained_transcript_turn_count=0,
            )

        kept_audio_turn_indexes = set(audio_turn_indexes[-normalized_limit:]) if normalized_limit else set()
        updated_history = [dict(item) for item in history]
        updated_turn_media = [
            TurnMedia(images=list(media.images), audios=list(media.audios))
            for media in turn_media
        ]
        updated_message_text = message_text
        retained_transcript_turn_count = 0
        updated_entries = dict(normalized_state.get("entries") or {})
        transcriber = CurrentModelAudioTranscriber(model, self.prompt_builder)

        for index in audio_turn_indexes:
            if index in kept_audio_turn_indexes:
                continue

            audio_path = updated_turn_media[index].audios[0]
            transcript_key = self._build_audio_fingerprint(audio_path)
            transcript = updated_entries.get(transcript_key)
            if not transcript:
                transcript = transcriber.transcribe(audio_path)
                updated_entries[transcript_key] = transcript

            updated_turn_media[index].audios = []
            if index < len(updated_history):
                updated_history[index]["content"] = self._append_transcript_block(
                    updated_history[index].get("content"),
                    transcript,
                )
            else:
                updated_message_text = self._append_transcript_block(updated_message_text, transcript)
            retained_transcript_turn_count += 1

        return AudioRetentionResult(
            message_text=updated_message_text,
            history=updated_history,
            turn_media=updated_turn_media,
            transcript_state=create_audio_transcript_state(updated_entries),
            retained_transcript_turn_count=retained_transcript_turn_count,
        )

    @staticmethod
    def _build_audio_fingerprint(audio_path: str) -> str:
        resolved_path = Path(audio_path).expanduser().resolve()
        file_stat = resolved_path.stat()
        fingerprint_source = f"{resolved_path}\n{file_stat.st_size}\n{file_stat.st_mtime_ns}"
        return hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()

    @staticmethod
    def _append_transcript_block(content: Any, transcript: str) -> str:
        normalized_content = str(content or "").strip()
        transcript_block = f"{AUDIO_TRANSCRIPT_BLOCK_LABEL}:\n{transcript.strip()}"
        if not normalized_content:
            return transcript_block
        return f"{normalized_content}\n\n{transcript_block}"
