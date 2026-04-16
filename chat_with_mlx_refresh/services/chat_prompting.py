from __future__ import annotations

from typing import Any

from ..model import BaseLocalModel, Message, MessageRole, MultimodalModel
from .chat_types import PreparedPrompt, TurnMedia, flatten_turn_media, turn_media_sequence_to_dicts


class ChatPromptBuilder:
    def prepare_prompt(
        self,
        model: BaseLocalModel,
        message_text: str,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> PreparedPrompt:
        current_message = Message(MessageRole.USER, message_text).to_dict()
        turn_media_dicts = turn_media_sequence_to_dicts(turn_media)
        formatted_prompt = model.format_chat_prompt(
            message=current_message,
            history=history,
            turn_media=turn_media_dicts,
        )
        flat_images, flat_audios = flatten_turn_media(turn_media)
        prepared_inputs = None
        if isinstance(model, MultimodalModel):
            prepared_inputs = model.prepare_prompt_inputs(
                formatted_prompt,
                images=flat_images,
                audios=flat_audios,
            )
            prompt_token_ids = list(model.extract_prompt_token_ids(prepared_inputs))
        elif callable(getattr(model, "encode_prompt_tokens", None)):
            prompt_token_ids = list(model.encode_prompt_tokens(formatted_prompt))
        else:
            prompt_token_ids = self._encode_text(self._get_text_tokenizer(model), formatted_prompt)

        return PreparedPrompt(
            formatted_prompt=formatted_prompt,
            prompt_token_ids=prompt_token_ids,
            prepared_inputs=prepared_inputs,
            flat_images=flat_images,
            flat_audios=flat_audios,
            turn_media_sequence=list(turn_media),
        )

    def count_prompt_tokens(
        self,
        model: BaseLocalModel,
        message_text: str,
        history: list[dict[str, Any]],
        turn_media: list[TurnMedia],
    ) -> tuple[int, PreparedPrompt]:
        prepared_prompt = self.prepare_prompt(
            model=model,
            message_text=message_text,
            history=history,
            turn_media=turn_media,
        )
        return len(prepared_prompt.prompt_token_ids), prepared_prompt

    @staticmethod
    def _get_text_tokenizer(model: BaseLocalModel):
        if isinstance(model, MultimodalModel):
            return model.processor.tokenizer if hasattr(model.processor, "tokenizer") else model.processor
        return getattr(model, "tokenizer", None) or getattr(model, "processor", None)

    @staticmethod
    def _encode_text(tokenizer: Any, text: str) -> list[int]:
        if tokenizer is None:
            raise RuntimeError("No tokenizer available for token counting.")
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
