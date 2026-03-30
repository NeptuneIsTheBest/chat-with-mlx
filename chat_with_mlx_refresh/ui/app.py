from __future__ import annotations

import gradio as gr

from ..context import AppContext
from ..services.chat import ChatService
from ..services.completion import CompletionService
from ..services.model_management import ModelManagementService
from ..services.runtime import RuntimeService
from .events import bind_events
from .layout import build_app_layout


def create_app(context: AppContext) -> gr.Blocks:
    runtime_service = RuntimeService(
        model_manager=context.model_manager,
        file_service=context.file_service,
        generation_stop_event=context.generation_stop_event,
    )
    model_management_service = ModelManagementService(context.model_manager)
    chat_service = ChatService(
        model_manager=context.model_manager,
        file_service=context.file_service,
        rag_service=context.rag_service,
        generation_stop_event=context.generation_stop_event,
    )
    completion_service = CompletionService(
        model_manager=context.model_manager,
        generation_stop_event=context.generation_stop_event,
    )

    ui = build_app_layout(
        runtime_service=runtime_service,
        model_management_service=model_management_service,
        rag_service=context.rag_service,
        chat_fn=chat_service.managed_chat_generator,
        completion_fn=completion_service.managed_completion_generator,
        bind_fn=lambda ui: bind_events(
            ui=ui,
            runtime_service=runtime_service,
            model_management_service=model_management_service,
            rag_service=context.rag_service,
        ),
    )
    return ui.app
