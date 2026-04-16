from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Iterator, Optional

from ..model import BaseLocalModel, ModelManager, MultimodalModel
from .async_stream import ThreadedGeneratorBridge
from .chat_preprocessing import ChatPreprocessor
from .chat_prompting import ChatPromptBuilder
from .chat_types import NormalizedTurn, turn_media_sequence_to_dicts
from .context_management import ContextManagementService
from .error_handling import raise_gradio_error
from .files import FileService
from .prompt_cache import PromptTokenRecorder, create_empty_prompt_cache_state
from .rag import RAGService
from .streaming import StreamSession


logger = logging.getLogger(__name__)


def ensure_model_has_chat_template(model: BaseLocalModel) -> None:
    if isinstance(model, MultimodalModel):
        return

    tokenizer_to_check = getattr(model, "tokenizer", None)
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
        self.prompt_builder = ChatPromptBuilder()
        self.preprocessor = ChatPreprocessor(file_service)
        self.context_management = ContextManagementService(prompt_builder=self.prompt_builder)

    def get_loaded_model(self) -> BaseLocalModel:
        model = self.model_manager.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    def _apply_rag_if_needed(
        self,
        message: NormalizedTurn,
        rag_enabled: bool,
        rag_n_results: int,
    ) -> NormalizedTurn:
        if not rag_enabled or not message.text:
            return message
        return NormalizedTurn(
            role=message.role,
            text=self.rag_service.enhance_message(message.text, rag_enabled, rag_n_results),
            attachments=list(message.attachments),
        )

    def reset_context_state(self, auto_manage_context: bool = True) -> tuple[dict[str, Any], str]:
        return self.context_management.reset_context_state(auto_manage_context)

    def reset_chat_state(
        self,
        auto_manage_context: bool = True,
        prompt_cache_state: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], str, dict[str, Any]]:
        self.model_manager.release_chat_cache_session(prompt_cache_state)
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
            normalized_message = self.preprocessor.normalize_incoming_message(message)
            normalized_message = self._apply_rag_if_needed(normalized_message, rag_enabled, rag_n_results)
            with self.model_manager.reserve_loaded_model() as model:
                ensure_model_has_chat_template(model)

                processed_message_text, processed_history_list, turn_media = self.preprocessor.prepare_model_inputs(
                    normalized_message,
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
                prepared_prompt = context_result.prepared_prompt
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
                    "formatted_prompt": prepared_prompt.formatted_prompt,
                    **sampling,
                }
                if prepared_prompt.prepared_inputs is not None:
                    response_args["prepared_inputs"] = prepared_prompt.prepared_inputs

                cache_plan = self.model_manager.prepare_chat_generation(
                    model=model,
                    prepared_prompt=prepared_prompt,
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
                    response_args["images"] = prepared_prompt.flat_images
                    response_args["audios"] = prepared_prompt.flat_audios
                    response_args["turn_media"] = turn_media_sequence_to_dicts(context_result.turn_media)

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
                transient_prompt_cache_state = self.model_manager.normalize_prompt_cache_state(prompt_cache_state)
                last_payload = None
                for payload in session:
                    last_payload = payload
                    yield payload, context_result.summary_state, context_result.status_text, transient_prompt_cache_state

                final_payload = session.final_messages or last_payload
                if final_payload is not None and final_payload != last_payload:
                    yield final_payload, context_result.summary_state, context_result.status_text, transient_prompt_cache_state

                if cache_plan is not None and not self.generation_stop_event.is_set():
                    updated_prompt_cache_state = self.model_manager.commit_chat_generation(
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
            raise_gradio_error(exc, logger=logger, action="stream chat responses")
