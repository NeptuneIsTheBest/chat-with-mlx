from __future__ import annotations

from ..services.chat import ChatService
from ..services.model_management import ModelManagementService
from ..services.rag import RAGService
from ..services.runtime import RuntimeService
from .components import AppUI


LOAD_CONCURRENCY_ID = "model-load"


def echo_value(value):
    return value


def bind_events(
    ui: AppUI,
    chat_service: ChatService,
    runtime_service: RuntimeService,
    model_management_service: ModelManagementService,
    rag_service: RAGService,
) -> None:
    ui.chat.default_system_prompt_button.click(
        fn=runtime_service.get_default_system_prompt,
        outputs=[ui.chat.system_prompt],
    ).then(
        fn=runtime_service.model_manager.set_custom_prompt,
        inputs=[ui.chat.system_prompt],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )
    ui.chat.system_prompt.change(
        fn=runtime_service.model_manager.set_custom_prompt,
        inputs=[ui.chat.system_prompt],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )
    ui.chat.auto_manage_context.change(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )
    ui.chat.chatbot.clear(
        fn=runtime_service.clear_cache,
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )

    ui.chat.model_selector.select(
        fn=echo_value,
        inputs=[ui.chat.model_selector],
        outputs=[ui.completion.model_selector],
    )
    ui.completion.model_selector.select(
        fn=echo_value,
        inputs=[ui.completion.model_selector],
        outputs=[ui.chat.model_selector],
    )

    ui.chat.load_button.click(
        fn=runtime_service.stop_generation,
        queue=False,
    ).then(
        fn=runtime_service.chat_load_model_callback,
        inputs=[ui.chat.model_selector],
        outputs=[
            ui.chat.model_status,
            ui.completion.model_status,
            ui.chat.system_prompt,
            ui.chat.memory,
            ui.completion.memory,
            ui.timer,
        ],
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id=LOAD_CONCURRENCY_ID,
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.chat.params["max_tokens"]],
        outputs=[ui.chat.params["max_tokens"]],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.completion.params["max_tokens"]],
        outputs=[ui.completion.params["max_tokens"]],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )

    ui.completion.load_button.click(
        fn=runtime_service.stop_generation,
        queue=False,
    ).then(
        fn=runtime_service.completion_load_model_callback,
        inputs=[ui.completion.model_selector],
        outputs=[
            ui.chat.model_status,
            ui.completion.model_status,
            ui.chat.system_prompt,
            ui.chat.memory,
            ui.completion.memory,
            ui.timer,
        ],
        trigger_mode="once",
        concurrency_limit=1,
        concurrency_id=LOAD_CONCURRENCY_ID,
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.chat.params["max_tokens"]],
        outputs=[ui.chat.params["max_tokens"]],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.completion.params["max_tokens"]],
        outputs=[ui.completion.params["max_tokens"]],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )

    ui.model_management.form["search_button"].click(
        fn=model_management_service.search_huggingface_models,
        inputs=[ui.model_management.form["search_query"]],
        outputs=[ui.model_management.form["search_results"]],
    )
    ui.model_management.form["search_results"].select(
        fn=model_management_service.auto_fill_model_info,
        inputs=[ui.model_management.form["search_results"]],
        outputs=[
            ui.model_management.form["model_name"],
            ui.model_management.form["mlx_repo"],
            ui.model_management.form["quantize"],
            ui.model_management.form["kv_cache_backend"],
            ui.model_management.form["turboquant_bits"],
        ],
    ).then(
        fn=model_management_service.update_turboquant_ui_state,
        inputs=[
            ui.model_management.form["kv_cache_backend"],
            ui.model_management.form["turboquant_bits"],
        ],
        outputs=[ui.model_management.form["turboquant_bits"]],
    ).then(
        fn=model_management_service.update_multimodal_ui_state,
        inputs=[
            ui.model_management.form["mlx_repo"],
            ui.model_management.form["multimodal_mode"],
            ui.model_management.form["multimodal_ability_override"],
        ],
        outputs=[
            ui.model_management.form["multimodal_ability_override"],
            ui.model_management.form["detected_capabilities"],
        ],
    )
    ui.model_management.form["mlx_repo"].change(
        fn=model_management_service.update_multimodal_ui_state,
        inputs=[
            ui.model_management.form["mlx_repo"],
            ui.model_management.form["multimodal_mode"],
            ui.model_management.form["multimodal_ability_override"],
        ],
        outputs=[
            ui.model_management.form["multimodal_ability_override"],
            ui.model_management.form["detected_capabilities"],
        ],
    )
    ui.model_management.form["multimodal_mode"].change(
        fn=model_management_service.update_multimodal_ui_state,
        inputs=[
            ui.model_management.form["mlx_repo"],
            ui.model_management.form["multimodal_mode"],
            ui.model_management.form["multimodal_ability_override"],
        ],
        outputs=[
            ui.model_management.form["multimodal_ability_override"],
            ui.model_management.form["detected_capabilities"],
        ],
    )
    ui.model_management.form["kv_cache_backend"].change(
        fn=model_management_service.update_turboquant_ui_state,
        inputs=[
            ui.model_management.form["kv_cache_backend"],
            ui.model_management.form["turboquant_bits"],
        ],
        outputs=[ui.model_management.form["turboquant_bits"]],
    )
    ui.model_management.form["add_button"].click(
        fn=model_management_service.add_model,
        inputs=[
            ui.model_management.form["model_name"],
            ui.model_management.form["mlx_repo"],
            ui.model_management.form["quantize"],
            ui.model_management.form["kv_cache_backend"],
            ui.model_management.form["turboquant_bits"],
            ui.model_management.form["default_language"],
            ui.model_management.form["system_prompt"],
            ui.model_management.form["multimodal_mode"],
            ui.model_management.form["multimodal_ability_override"],
        ],
    ).then(
        fn=model_management_service.update_model_management_models_list,
        outputs=[ui.model_management.model_list],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.chat.model_selector],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.completion.model_selector],
    ).then(
        fn=model_management_service.update_delete_model_selector_choices,
        outputs=[ui.model_management.form["delete_model_selector"]],
    )
    ui.model_management.form["delete_button"].click(
        fn=model_management_service.delete_model,
        inputs=[
            ui.model_management.form["delete_model_selector"],
            ui.model_management.form["delete_files_checkbox"],
        ],
        outputs=[ui.model_management.form["delete_status"]],
    ).then(
        fn=model_management_service.update_model_management_models_list,
        outputs=[ui.model_management.model_list],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.chat.model_selector],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.completion.model_selector],
    ).then(
        fn=model_management_service.update_delete_model_selector_choices,
        outputs=[ui.model_management.form["delete_model_selector"]],
    ).then(
        fn=runtime_service.get_runtime_snapshot,
        outputs=[
            ui.chat.model_status,
            ui.completion.model_status,
            ui.chat.system_prompt,
            ui.chat.memory,
            ui.completion.memory,
            ui.timer,
        ],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.chat.params["max_tokens"]],
        outputs=[ui.chat.params["max_tokens"]],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.completion.params["max_tokens"]],
        outputs=[ui.completion.params["max_tokens"]],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )

    ui.chat.rag_form["rag_enabled"].change(
        fn=rag_service.toggle_enabled,
        inputs=[ui.chat.rag_form["rag_enabled"]],
        outputs=[ui.chat.rag_form["upload_status"]],
    )
    ui.chat.rag_form["upload_button"].click(
        fn=lambda files: rag_service.upload_and_index_files(files, runtime_service.file_service),
        inputs=[ui.chat.rag_form["file_upload"]],
        outputs=[ui.chat.rag_form["upload_status"], ui.chat.rag_form["rag_status"]],
    )
    ui.chat.rag_form["clear_button"].click(
        fn=rag_service.clear_index_with_status,
        outputs=[ui.chat.rag_form["upload_status"], ui.chat.rag_form["rag_status"]],
    )
    ui.chat.rag_form["update_params_button"].click(
        fn=rag_service.update_ui_parameters,
        inputs=[
            ui.chat.rag_params["chunk_size"],
            ui.chat.rag_params["chunk_overlap"],
            ui.chat.rag_params["n_results"],
            ui.chat.rag_params["similarity_threshold"],
        ],
        outputs=[ui.chat.rag_form["params_status"]],
    )

    ui.app.load(
        fn=model_management_service.update_model_management_models_list,
        outputs=[ui.model_management.model_list],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.chat.model_selector],
    ).then(
        fn=model_management_service.update_model_selector_choices,
        outputs=[ui.completion.model_selector],
    ).then(
        fn=runtime_service.get_runtime_snapshot,
        outputs=[
            ui.chat.model_status,
            ui.completion.model_status,
            ui.chat.system_prompt,
            ui.chat.memory,
            ui.completion.memory,
            ui.timer,
        ],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.chat.params["max_tokens"]],
        outputs=[ui.chat.params["max_tokens"]],
    ).then(
        fn=runtime_service.update_model_max_length,
        inputs=[ui.completion.params["max_tokens"]],
        outputs=[ui.completion.params["max_tokens"]],
    ).then(
        fn=rag_service.get_status_text,
        outputs=[ui.chat.rag_form["rag_status"]],
    ).then(
        fn=chat_service.reset_chat_state,
        inputs=[ui.chat.auto_manage_context, ui.chat.prompt_cache_state],
        outputs=[ui.chat.context_summary_state, ui.chat.context_status, ui.chat.prompt_cache_state],
    )
