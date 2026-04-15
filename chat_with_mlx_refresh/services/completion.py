from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Iterator, Optional

from ..model import BaseLocalModel, ModelManager
from .async_stream import ThreadedGeneratorBridge
from .error_handling import raise_gradio_error
from .streaming import CHATML_CONTROL_TOKENS, CHATML_STOP_TOKENS, is_supported_chunk_type, parse_chunk_text
from .structured_output import IncrementalTextSanitizer


logger = logging.getLogger(__name__)
COMPLETION_STOP_TOKENS = CHATML_STOP_TOKENS + ["<|im_start|>", "<|system|>", "<|user|>", "<|assistant|>"]


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

    def handle_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 0.9,
        min_p: float = 0.0,
        max_tokens: int = 512,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 1.5,
        stream: bool = True,
    ) -> Iterator[str] | Optional[str]:
        try:
            with self.model_manager.reserve_loaded_model() as model:
                params = {
                    "stream": stream,
                    "temperature": float(temperature),
                    "top_k": top_k,
                    "top_p": float(top_p),
                    "min_p": min_p,
                    "max_tokens": max_tokens,
                    "repetition_penalty": float(repetition_penalty),
                    "presence_penalty": float(presence_penalty),
                }

                if not stream:
                    completion_result = model.generate_completion(prompt=prompt, **params)
                    completion_text = parse_chunk_text(completion_result) or str(completion_result)
                    sanitizer = IncrementalTextSanitizer(
                        control_tokens=CHATML_CONTROL_TOKENS,
                        stop_tokens=COMPLETION_STOP_TOKENS,
                        eos_token=self._get_eos_token(model),
                    )
                    sanitized = sanitizer.feed(completion_text)
                    tail = sanitizer.finalize()
                    return f"{prompt}{sanitized.text}{tail.text}"

                response_text = prompt
                sanitizer = IncrementalTextSanitizer(
                    control_tokens=CHATML_CONTROL_TOKENS,
                    stop_tokens=COMPLETION_STOP_TOKENS,
                    eos_token=self._get_eos_token(model),
                )
                for chunk in model.generate_completion(prompt=prompt, **params):
                    if self.generation_stop_event.is_set():
                        break

                    chunk_text = parse_chunk_text(chunk)
                    if not chunk_text:
                        if not is_supported_chunk_type(chunk):
                            logger.warning("Unexpected chunk type from model: %s", type(chunk))
                        continue

                    sanitized = sanitizer.feed(chunk_text)
                    if not sanitized.text:
                        if sanitized.stop:
                            break
                        continue

                    response_text += sanitized.text
                    yield response_text

                    if sanitized.stop:
                        break

                tail = sanitizer.finalize()
                if tail.text:
                    response_text += tail.text
                    yield response_text
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
        presence_penalty: float = 1.5,
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
                presence_penalty=presence_penalty,
                stream=stream,
            )
            await asyncio.to_thread(self.model_manager.close_active_generator, clear_runtime_cache=False)
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
            raise_gradio_error(exc, logger=logger, action="stream completion responses")
