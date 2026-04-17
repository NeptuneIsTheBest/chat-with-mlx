from __future__ import annotations

import logging
from typing import Optional

import gradio as gr
from huggingface_hub import HfApi
from pandas import DataFrame

from ..language import get_text
from ..model import ModelManager
from ..model_config import ModelConfigStore
from .error_handling import get_exception_message, gradio_error_boundary, log_service_exception


logger = logging.getLogger(__name__)


class ModelManagementService:
    AVAILABLE_MULTIMODAL_ABILITIES = ("vision", "audio")
    CAPABILITY_MODE_TEXT_ONLY = ModelConfigStore.CAPABILITY_MODE_TEXT_ONLY
    CAPABILITY_MODE_AUTO_DETECT = ModelConfigStore.CAPABILITY_MODE_AUTO_DETECT
    CAPABILITY_MODE_MANUAL_OVERRIDE = ModelConfigStore.CAPABILITY_MODE_MANUAL_OVERRIDE
    KV_CACHE_BACKEND_DEFAULT = ModelConfigStore.KV_CACHE_BACKEND_DEFAULT
    KV_CACHE_BACKEND_TURBOQUANT = ModelConfigStore.KV_CACHE_BACKEND_TURBOQUANT

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def get_model_list(self) -> list[str]:
        return self.model_manager.list_model_names()

    def get_capability_mode_choices(self) -> list[str]:
        return [
            self.CAPABILITY_MODE_TEXT_ONLY,
            self.CAPABILITY_MODE_AUTO_DETECT,
            self.CAPABILITY_MODE_MANUAL_OVERRIDE,
        ]

    def get_kv_cache_backend_choices(self) -> list[str]:
        return [
            self.KV_CACHE_BACKEND_DEFAULT,
            self.KV_CACHE_BACKEND_TURBOQUANT,
        ]

    @staticmethod
    def get_turboquant_bits_choices() -> list[int]:
        return sorted(ModelConfigStore.VALID_TURBOQUANT_BITS)

    @staticmethod
    def _is_valid_repo(mlx_repo: str) -> bool:
        normalized_repo = (mlx_repo or "").strip()
        return len(normalized_repo.split("/")) == 2 and all(normalized_repo.split("/"))

    def _normalize_capability_mode(self, multimodal_mode: Optional[str]) -> str:
        if multimodal_mode in self.get_capability_mode_choices():
            return multimodal_mode
        return self.CAPABILITY_MODE_TEXT_ONLY

    def _normalize_kv_cache_backend(self, kv_cache_backend: Optional[str]) -> str:
        normalized = ModelConfigStore.normalize_kv_cache_backend(kv_cache_backend)
        if normalized in self.get_kv_cache_backend_choices():
            return normalized
        return self.KV_CACHE_BACKEND_DEFAULT

    @staticmethod
    def _normalize_turboquant_bits(turboquant_bits: Optional[int]) -> int:
        normalized = ModelConfigStore.normalize_turboquant_bits(turboquant_bits)
        return normalized if normalized is not None else 4

    @gradio_error_boundary("search HuggingFace models", logger)
    def search_huggingface_models(self, query: str) -> DataFrame:
        if not query:
            return DataFrame(columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))

        api = HfApi()
        models = api.list_models(search=query, filter="mlx", sort="likes", limit=100)
        sorted_models = sorted(models, key=lambda model: model.likes, reverse=True)[:50]
        data = [[model.modelId, model.likes, model.downloads] for model in sorted_models]
        return DataFrame(data, columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))

    def _detect_model_capabilities(self, mlx_repo: str) -> list[str]:
        normalized_repo = (mlx_repo or "").strip()
        return self.model_manager.config_store.detect_multimodal_abilities_from_repo(normalized_repo)

    def update_multimodal_ui_state(
        self,
        mlx_repo: str,
        multimodal_mode: Optional[str],
        multimodal_ability_override: Optional[list[str]],
    ):
        normalized_mode = self._normalize_capability_mode(multimodal_mode)
        current_override = list(multimodal_ability_override or [])

        if normalized_mode == self.CAPABILITY_MODE_TEXT_ONLY:
            return (
                gr.update(visible=False, value=current_override),
                "Text only mode selected. The model will be loaded as a text model.",
            )

        if not self._is_valid_repo(mlx_repo):
            if normalized_mode == self.CAPABILITY_MODE_MANUAL_OVERRIDE:
                return (
                    gr.update(visible=True, value=current_override),
                    "Manual override enabled. Select the abilities this model should use.",
                )
            return (
                gr.update(visible=False, value=current_override),
                "Auto detect will run after you enter a valid repository.",
            )

        try:
            abilities = self._detect_model_capabilities(mlx_repo)
            formatted_abilities = ModelConfigStore.format_multimodal_abilities(abilities)
        except Exception as exc:
            message = get_exception_message(exc)
            if normalized_mode == self.CAPABILITY_MODE_MANUAL_OVERRIDE:
                return (
                    gr.update(visible=True, value=current_override),
                    f"Manual override enabled. Detection failed: {message}",
                )
            return (
                gr.update(visible=False, value=current_override),
                f"Auto detect failed: {message}. Switch to Text only or Manual override.",
            )

        if normalized_mode == self.CAPABILITY_MODE_MANUAL_OVERRIDE:
            override_value = current_override or abilities
            return (
                gr.update(visible=True, value=override_value),
                f"Manual override enabled. Suggested abilities: {formatted_abilities}.",
            )

        return (
            gr.update(visible=False, value=current_override or abilities),
            f"Auto-detected: {formatted_abilities}",
        )

    def detect_model_capabilities_state(self, mlx_repo: str):
        normalized_repo = (mlx_repo or "").strip()
        if len(normalized_repo.split("/")) != 2 or not all(normalized_repo.split("/")):
            return gr.update(value=[]), ""

        try:
            abilities = self._detect_model_capabilities(normalized_repo)
            formatted_abilities = ModelConfigStore.format_multimodal_abilities(abilities)
            return gr.update(value=abilities), f"Auto-detected: {formatted_abilities}"
        except Exception as exc:
            log_service_exception(
                logger,
                f"detect multimodal abilities for {normalized_repo}",
                exc,
                level=logging.WARNING,
                include_traceback=False,
            )
            return gr.update(value=[]), f"Detection failed: {get_exception_message(exc)}"

    def auto_fill_model_info(self, data_frame: DataFrame, evt: gr.SelectData = None):
        if isinstance(data_frame, gr.SelectData):
            data_frame, evt = evt, data_frame

        if evt is None or not isinstance(data_frame, DataFrame):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        if evt.index[0] < 0 or evt.index[0] >= len(data_frame):
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

        model_id = str(data_frame.iloc[evt.index[0], 0])
        model_name = model_id.split("/")[-1]

        quantize = "None"
        lower_name = model_name.lower()
        if "5bit" in lower_name or "q5" in lower_name:
            quantize = "5bit"
        elif "4bit" in lower_name or "q4" in lower_name:
            quantize = "4bit"
        elif "8bit" in lower_name or "q8" in lower_name:
            quantize = "8bit"
        elif "6bit" in lower_name or "q6" in lower_name:
            quantize = "6bit"
        elif "3bit" in lower_name or "q3" in lower_name:
            quantize = "3bit"
        elif "2bit" in lower_name or "q2" in lower_name:
            quantize = "2bit"
        elif "bf16" in lower_name:
            quantize = "bf16"
        elif "bf32" in lower_name:
            quantize = "bf32"

        kv_cache_backend = self.KV_CACHE_BACKEND_TURBOQUANT if "optiq" in lower_name else self.KV_CACHE_BACKEND_DEFAULT
        turboquant_bits = 4

        return model_name, model_id, quantize, kv_cache_backend, turboquant_bits

    def update_turboquant_ui_state(
        self,
        kv_cache_backend: Optional[str],
        turboquant_bits: Optional[int],
    ):
        normalized_backend = self._normalize_kv_cache_backend(kv_cache_backend)
        normalized_bits = self._normalize_turboquant_bits(turboquant_bits)
        return gr.update(
            visible=normalized_backend == self.KV_CACHE_BACKEND_TURBOQUANT,
            value=normalized_bits,
        )

    def update_model_management_models_list(self) -> DataFrame:
        return DataFrame({get_text("Page.ModelManagement.Dataframe.model_list.headers"): self.get_model_list()})

    def update_select_model_dropdown_value(self) -> Optional[str]:
        loaded_config = self.model_manager.get_loaded_model_config()
        if loaded_config:
            return loaded_config.resolved_display_name

        model_list = self.get_model_list()
        return model_list[0] if model_list else None

    def update_model_selector_choices(self):
        return gr.update(choices=self.get_model_list(), value=self.update_select_model_dropdown_value())

    def update_delete_model_selector_choices(self, current_value: Optional[str] = None):
        model_list = self.get_model_list()
        value = current_value if current_value in model_list else None
        return gr.update(choices=model_list, value=value)

    @gradio_error_boundary("add a model configuration", logger)
    def add_model(
        self,
        model_name: Optional[str],
        mlx_repo: str,
        quantize: str,
        kv_cache_backend: Optional[str],
        turboquant_bits: Optional[int],
        default_language: str,
        default_system_prompt: Optional[str],
        multimodal_mode: Optional[str],
        multimodal_ability_override: Optional[list[str]],
    ) -> None:
        self.model_manager.config_store.add_config(
            mlx_repo=mlx_repo,
            model_name=model_name,
            quantize=quantize,
            kv_cache_backend=self._normalize_kv_cache_backend(kv_cache_backend),
            turboquant_bits=self._normalize_turboquant_bits(turboquant_bits),
            default_language=default_language,
            system_prompt=default_system_prompt,
            multimodal_mode=self._normalize_capability_mode(multimodal_mode),
            multimodal_ability_override=multimodal_ability_override,
        )
        self.model_manager.refresh_model_configs()

    @gradio_error_boundary("delete a model configuration", logger)
    def delete_model(self, model_name: str, delete_files: bool) -> str:
        if not model_name:
            raise ValueError(get_text("Page.ModelManagement.DeleteModelBlock.Messages.no_model_selected"))

        model_configs = self.model_manager.refresh_model_configs()
        model_config = model_configs.get(model_name)
        if model_config is None:
            raise RuntimeError(f"Model '{model_name}' not found")

        cleanup_warnings: list[str] = []
        loaded_config = self.model_manager.get_loaded_model_config()
        if loaded_config and loaded_config.resolved_display_name == model_name:
            cleanup_warnings = self.model_manager.close_model()

        self.model_manager.config_store.delete_config(model_config, delete_model_files=delete_files)
        self.model_manager.refresh_model_configs()

        if delete_files:
            message = get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_and_files_deleted").format(
                model_name
            )
        else:
            message = get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_deleted").format(model_name)

        if cleanup_warnings:
            return f"{message} {get_text('Page.ModelManagement.DeleteModelBlock.Messages.cleanup_warning_suffix')}"
        return message
