from __future__ import annotations

import logging
from typing import Any, Optional, Union

import gradio as gr

from ..language import get_text
from ..model import BaseLocalModel, ModelManager, MultimodalModel, TextModel
from .error_handling import gradio_error_boundary, log_service_exception
from .files import FileService


logger = logging.getLogger(__name__)


class RuntimeService:
    def __init__(self, model_manager: ModelManager, file_service: FileService, generation_stop_event: Any) -> None:
        self.model_manager = model_manager
        self.file_service = file_service
        self.generation_stop_event = generation_stop_event

    def get_loaded_model(self) -> BaseLocalModel:
        model = self.model_manager.get_loaded_model()
        if model is None:
            raise RuntimeError("No model loaded.")
        return model

    def get_load_model_status(self) -> str:
        loaded_config = self.model_manager.get_loaded_model_config()
        if loaded_config:
            return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value").format(
                loaded_config.resolved_display_name
            )
        return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")

    def get_default_system_prompt_value(self) -> str:
        return self.model_manager.get_system_prompt(default=True) or ""

    def get_memory_timer_update(self):
        return gr.Timer(value=1, active=self.model_manager.has_loaded_model(), render=False)

    @staticmethod
    def _format_memory_value(memory_bytes: Any) -> str:
        if not isinstance(memory_bytes, (int, float)):
            return "N/A"
        return f"{memory_bytes / 1024 ** 3:.2f} GB"

    def load_model(self, model_name: str):
        self.model_manager.load_model(model_name)
        return self.get_runtime_snapshot()

    def stop_generation(self) -> None:
        self.generation_stop_event.set()

    @gradio_error_boundary("load the selected model for chat", logger)
    def chat_load_model_callback(self, model_name: str):
        self.stop_generation()
        return self.load_model(model_name)

    @gradio_error_boundary("load the selected model for completion", logger)
    def completion_load_model_callback(self, model_name: str):
        self.stop_generation()
        return self.load_model(model_name)

    @gradio_error_boundary("read the default system prompt", logger)
    def get_default_system_prompt(self) -> Optional[str]:
        return self.get_default_system_prompt_value()

    @staticmethod
    def update_slider_config(
        slider_new_min: Union[int, float],
        slider_new_max: Union[int, float],
        slider_value: Union[int, float, None],
    ):
        if slider_new_min is None or slider_new_max is None or slider_new_min > slider_new_max:
            return gr.update()

        value_to_set = slider_value if slider_value is not None else (slider_new_min + slider_new_max) / 2
        if isinstance(slider_new_min, int) and isinstance(slider_new_max, int):
            value_to_set = int(value_to_set)

        value_to_set = max(slider_new_min, min(slider_new_max, value_to_set))
        return gr.update(minimum=slider_new_min, maximum=slider_new_max, value=value_to_set)

    def update_model_max_length(self, slider_value: Union[int, float, None]):
        default_max_length = 32768
        try:
            with self.model_manager.reserve_loaded_model() as model:
                max_length = default_max_length
                if isinstance(model, (TextModel, MultimodalModel)):
                    max_length = model.max_position_embeddings or default_max_length
                    if max_length <= 0:
                        max_length = default_max_length
                return self.update_slider_config(1, max_length, slider_value)
        except RuntimeError:
            return self.update_slider_config(1, default_max_length, slider_value)
        except Exception as exc:
            log_service_exception(
                logger,
                "resolve the loaded model before updating max token limits",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return self.update_slider_config(1, default_max_length, slider_value)

    def get_memory_usage(self) -> str:
        try:
            memory_usage_bytes = 0 if not self.model_manager.has_loaded_model() else self.model_manager.get_system_memory_usage()
            total_memory_bytes = self.model_manager.get_device_info()["memory_size"]
            return f"{self._format_memory_value(memory_usage_bytes)} | {self._format_memory_value(total_memory_bytes)}"
        except Exception as exc:
            log_service_exception(
                logger,
                "read runtime memory usage",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return "N/A | N/A"

    def update_all_memory_usage(self):
        memory_usage = self.get_memory_usage()
        try:
            timer = self.get_memory_timer_update()
        except Exception as exc:
            log_service_exception(
                logger,
                "refresh the runtime memory timer",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            timer = gr.Timer(value=1, active=False, render=False)
        return [memory_usage, memory_usage, timer]

    def get_runtime_snapshot(self):
        try:
            status = self.get_load_model_status()
        except Exception as exc:
            log_service_exception(
                logger,
                "read the model load status",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            status = get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")
        try:
            system_prompt = self.get_default_system_prompt_value()
        except Exception as exc:
            log_service_exception(
                logger,
                "read the default system prompt value",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            system_prompt = ""
        memory_usage = self.get_memory_usage()
        try:
            timer = self.get_memory_timer_update()
        except Exception as exc:
            log_service_exception(
                logger,
                "build the runtime memory timer",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            timer = gr.Timer(value=1, active=False, render=False)
        return [status, status, system_prompt, memory_usage, memory_usage, timer]

    @gradio_error_boundary("clear the active runtime cache", logger)
    def clear_cache(self) -> None:
        self.stop_generation()
        self.file_service.clear()
        self.model_manager.close_active_generator()

    def close(self) -> None:
        self.model_manager.close_model()
