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

    def __init__(self, model_manager: ModelManager) -> None:
        self.model_manager = model_manager

    def get_model_list(self) -> list[str]:
        return self.model_manager.list_model_names()

    @gradio_error_boundary("search HuggingFace models", logger)
    def search_huggingface_models(self, query: str) -> DataFrame:
        if not query:
            return DataFrame(columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))

        api = HfApi()
        models = api.list_models(search=query, filter="mlx", sort="likes", limit=100)
        sorted_models = sorted(models, key=lambda model: model.likes, reverse=True)[:50]
        data = [[model.modelId, model.likes, model.downloads] for model in sorted_models]
        return DataFrame(data, columns=get_text("Page.ModelManagement.Dataframe.search_results.headers"))

    def detect_model_capabilities_state(self, mlx_repo: str):
        normalized_repo = (mlx_repo or "").strip()
        if len(normalized_repo.split("/")) != 2 or not all(normalized_repo.split("/")):
            return gr.update(value=[]), ""

        try:
            abilities = self.model_manager.config_store.detect_multimodal_abilities_from_repo(normalized_repo)
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

    def auto_fill_model_info(self, evt: gr.SelectData, data_frame: DataFrame):
        if evt.index[0] < 0 or evt.index[0] >= len(data_frame):
            return gr.update(), gr.update(), gr.update(), gr.update(value=[]), gr.update(value="")

        model_id = data_frame.iloc[evt.index[0]]["Model ID"]
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

        multimodal_override, detected_capabilities = self.detect_model_capabilities_state(model_id)
        return model_name, model_id, quantize, multimodal_override, detected_capabilities

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

    def update_delete_model_selector_choices(self):
        return gr.update(choices=self.get_model_list())

    @gradio_error_boundary("add a model configuration", logger)
    def add_model(
        self,
        model_name: Optional[str],
        mlx_repo: str,
        quantize: str,
        default_language: str,
        default_system_prompt: Optional[str],
        multimodal_ability_override: Optional[list[str]],
    ) -> None:
        self.model_manager.config_store.add_config(
            mlx_repo=mlx_repo,
            model_name=model_name,
            quantize=quantize,
            default_language=default_language,
            system_prompt=default_system_prompt,
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

        loaded_config = self.model_manager.get_loaded_model_config()
        if loaded_config and loaded_config.resolved_display_name == model_name:
            self.model_manager.close_model()

        self.model_manager.config_store.delete_config(model_config, delete_model_files=delete_files)
        self.model_manager.refresh_model_configs()

        if delete_files:
            return get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_and_files_deleted").format(
                model_name
            )
        return get_text("Page.ModelManagement.DeleteModelBlock.Messages.config_deleted").format(model_name)
