from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Iterator, Optional

from ..model import BaseLocalModel, ModelManager
from .async_stream import ThreadedGeneratorBridge
from .error_handling import raise_gradio_error
from .streaming import (
    CHATML_CONTROL_TOKENS,
    CHATML_STOP_TOKENS,
    filter_chatml_tokens_and_stop,
    is_supported_chunk_type,
    merge_with_partial_buffer,
    parse_chunk_text,
)


logger = logging.getLogger(__name__)
COMPLETION_STOP_TOKENS = CHATML_STOP_TOKENS + ["<|im_start|>", "<|system|>", "<|user|>", "<|assistant|>"]


def _build_partial_prefixes(tokens: list[str]) -> list[str]:
    prefixes = {token[:index] for token in tokens for index in range(1, len(token))}
    return sorted(prefixes, key=len)


COMPLETION_PARTIAL_PREFIXES = _build_partial_prefixes(CHATML_CONTROL_TOKENS)


class CompletionService:
    def __init__(self, model_manager: ModelManager, generation_stop_event: Any) -> None:
        self.model_manager = model_manager
        self.generation_stop_event = generation_stop_event

    def _get_loaded_model(self):
        model = self.model_manager.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    @staticmethod
    def _get_eos_token(model: BaseLocalModel) -> Optional[str]:
        if getattr(model, "tokenizer", None) is not None:
            return model.tokenizer.eos_token
        if getattr(model, "processor", None) and getattr(model.processor, "tokenizer", None):
            return model.processor.tokenizer.eos_token
        return None

    @staticmethod
    def _sanitize_completion_text(text: str, eos_token: Optional[str]) -> tuple[str, bool]:
        eos_hit = False
        if eos_token and eos_token in text:
            text = text.split(eos_token)[0]
            eos_hit = True

        filtered, should_stop = filter_chatml_tokens_and_stop(
            text,
            CHATML_CONTROL_TOKENS,
            COMPLETION_STOP_TOKENS,
        )
        return filtered, eos_hit or should_stop

    def handle_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        stream: bool = True,
    ) -> Iterator[str] | Optional[str]:
        try:
            model = self._get_loaded_model()

            params = {
                "stream": stream,
                "temperature": float(temperature),
                "top_k": top_k,
                "top_p": float(top_p),
                "min_p": min_p,
                "max_tokens": max_tokens,
                "repetition_penalty": float(repetition_penalty),
            }

            if not stream:
                completion_result = model.generate_completion(prompt=prompt, **params)
                completion_text = parse_chunk_text(completion_result) or str(completion_result)
                completion_text, _ = self._sanitize_completion_text(completion_text, self._get_eos_token(model))
                return f"{prompt}{completion_text}"

            response_parts = [prompt]
            eos_token = self._get_eos_token(model)
            chunk_buffer = ""
            for chunk in model.generate_completion(prompt=prompt, **params):
                if self.generation_stop_event.is_set():
                    break

                chunk_text = parse_chunk_text(chunk)
                if not chunk_text:
                    if not is_supported_chunk_type(chunk):
                        logger.warning("Unexpected chunk type from model: %s", type(chunk))
                    continue

                chunk_text, chunk_buffer = merge_with_partial_buffer(
                    chunk_buffer,
                    chunk_text,
                    COMPLETION_PARTIAL_PREFIXES,
                )
                if not chunk_text:
                    continue

                chunk_text, should_stop = self._sanitize_completion_text(chunk_text, eos_token)
                if chunk_text:
                    response_parts.append(chunk_text)
                    yield "".join(response_parts)

                if should_stop:
                    break
            return None
        except Exception as exc:
            raise_gradio_error(exc, logger=logger, action="handle completion requests")

    async def managed_completion_generator(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        stream: bool = True,
    ) -> AsyncIterator[str]:
        try:
            generator = self.handle_completion(
                prompt=prompt,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
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
            raise_gradio_error(exc, logger=logger, action="stream completion responses")
