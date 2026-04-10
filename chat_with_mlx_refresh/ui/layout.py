from __future__ import annotations

import gradio as gr
from gradio.i18n import I18nData

from ..language import get_text
from ..services.context_management import create_empty_context_summary_state, get_default_context_status
from ..services.model_management import ModelManagementService
from ..services.prompt_cache import create_empty_prompt_cache_state, delete_prompt_cache_state
from ..services.rag import RAGService
from ..services.runtime import RuntimeService
from .components import AppUI, ChatUI, CompletionUI, ModelManagementUI, create_generation_params, create_model_controls, create_rag_params, create_textbox


def build_app_layout(
    runtime_service: RuntimeService,
    model_management_service: ModelManagementService,
    rag_service: RAGService,
    chat_fn,
    completion_fn,
    bind_fn=None,
) -> AppUI:
    with gr.Blocks(fill_height=True, fill_width=True, title="Chat with MLX") as app:
        timer = gr.Timer(value=1, active=runtime_service.model_manager.has_loaded_model())

        chat_memory, chat_model_selector, chat_model_status, chat_load_button = create_model_controls(
            model_management_service.get_model_list(),
            runtime_service.get_load_model_status,
        )
        chat_params = create_generation_params()
        rag_params = create_rag_params(rag_service.get_parameter_tuple())
        chat_system_prompt_textbox = gr.Textbox(
            label=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.label"),
            placeholder=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.placeholder"),
            value=runtime_service.model_manager.get_system_prompt,
            lines=3,
            max_lines=5,
            buttons=["copy"],
            render=False,
            scale=9,
        )
        chat_default_system_prompt_button = gr.Button(
            value=get_text("Page.Chat.ChatSystemPromptBlock.Button.default_system_prompt.value"),
            render=False,
            scale=1,
        )
        chat_auto_manage_context_checkbox = gr.Checkbox(
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Checkbox.auto_manage_context.label"),
            value=True,
            interactive=True,
            render=False,
        )
        chat_context_status_textbox = gr.Textbox(
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Textbox.context_status.label"),
            value=get_default_context_status(),
            interactive=False,
            render=False,
            lines=2,
        )
        chat_context_summary_state = gr.State(value=create_empty_context_summary_state())
        chat_prompt_cache_state = gr.State(
            value=create_empty_prompt_cache_state(),
            delete_callback=delete_prompt_cache_state,
        )
        chat_rag_form = {
            "rag_enabled": gr.Checkbox(
                label=get_text("Page.Chat.Accordion.RAGSetting.Checkbox.rag_enabled.label"),
                value=rag_service.is_enabled,
                interactive=True,
                render=False,
            ),
            "file_upload": gr.File(
                label=get_text("Page.Chat.Accordion.RAGSetting.File.file_upload.label"),
                file_count="multiple",
                file_types=[".txt", ".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".md", ".csv"],
                render=False,
            ),
            "upload_button": gr.Button(
                value=get_text("Page.Chat.Accordion.RAGSetting.Button.upload.value"),
                render=False,
            ),
            "clear_button": gr.Button(
                value=get_text("Page.Chat.Accordion.RAGSetting.Button.clear.value"),
                render=False,
            ),
            "upload_status": create_textbox(
                "Page.Chat.Accordion.RAGSetting.Textbox.upload_status.label",
                interactive=False,
                render=False,
            ),
            "rag_status": create_textbox(
                "Page.Chat.Accordion.RAGSetting.Textbox.rag_status.label",
                interactive=False,
                render=False,
            ),
            "update_params_button": gr.Button(
                value=get_text("Page.Chat.Accordion.RAGSetting.Button.update_params.value"),
                render=False,
            ),
            "params_status": create_textbox(
                "Page.Chat.Accordion.RAGSetting.Textbox.params_status.label",
                interactive=False,
                render=False,
            ),
        }

        completion_memory, completion_model_selector, completion_model_status, completion_load_button = create_model_controls(
            model_management_service.get_model_list(),
            runtime_service.get_load_model_status,
        )
        completion_params = create_generation_params()

        local_model_form = {
            "search_query": gr.Textbox(
                label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.search_query.label"),
                placeholder=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.search_query.placeholder"),
                render=False,
            ),
            "search_button": gr.Button(
                value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.search.value"),
                render=False,
            ),
            "search_results": gr.Dataframe(
                headers=get_text("Page.ModelManagement.Dataframe.search_results.headers"),
                interactive=False,
                render=False,
            ),
            "model_name": create_textbox(
                "Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.label",
                "Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.placeholder",
            ),
            "mlx_repo": create_textbox(
                "Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.label",
                "Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.placeholder",
            ),
            "quantize": gr.Dropdown(
                label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.quantize.label"),
                choices=["None", "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "bf16", "bf32"],
                value="None",
                interactive=True,
                render=False,
            ),
            "default_language": gr.Dropdown(
                label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.default_language.label"),
                choices=["multi"],
                value="multi",
                interactive=True,
                render=False,
            ),
            "system_prompt": create_textbox("Page.ModelManagement.AddLocalModelBlock.Textbox.default_system_prompt.label"),
            "multimodal_mode": gr.Dropdown(
                label="Capability Mode",
                choices=model_management_service.get_capability_mode_choices(),
                value=ModelManagementService.CAPABILITY_MODE_TEXT_ONLY,
                interactive=True,
                render=False,
            ),
            "multimodal_ability_override": gr.CheckboxGroup(
                label=get_text("Page.ModelManagement.AddLocalModelBlock.CheckboxGroup.multimodal_ability_override.label"),
                choices=["vision", "audio"],
                value=[],
                interactive=True,
                render=False,
                visible=False,
            ),
            "detected_capabilities": gr.Textbox(
                label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.detected_capabilities.label"),
                interactive=False,
                render=False,
            ),
            "add_button": gr.Button(
                value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.add.value"),
                render=False,
            ),
            "delete_model_selector": gr.Dropdown(
                label=get_text("Page.ModelManagement.DeleteModelBlock.Dropdown.model_selector.label"),
                choices=model_management_service.get_model_list(),
                value=None,
                interactive=True,
                render=False,
            ),
            "delete_files_checkbox": gr.Checkbox(
                label=get_text("Page.ModelManagement.DeleteModelBlock.Checkbox.delete_files.label"),
                value=False,
                interactive=True,
                render=False,
            ),
            "delete_button": gr.Button(
                value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.delete.value"),
                variant="stop",
                render=False,
            ),
            "delete_status": gr.Textbox(
                label=get_text("Page.ModelManagement.DeleteModelBlock.Textbox.delete_status.label"),
                interactive=False,
                render=False,
            ),
        }
        model_list = gr.Dataframe(
            headers=[get_text("Page.ModelManagement.Dataframe.model_list.headers")],
            value=model_management_service.update_model_management_models_list(),
            datatype="str",
            row_count=5,
            render=False,
            interactive=False,
        )

        timer.tick(fn=runtime_service.update_all_memory_usage, outputs=[chat_memory, completion_memory, timer])
        gr.HTML("<h2>Chat with MLX</h2>")

        with gr.Tab(get_text("Tab.chat")):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        chat_memory.render()

                    with gr.Row():
                        gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")
                        chat_model_selector.render()
                        chat_model_status.render()
                        chat_load_button.render()

                    with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                        with gr.Group():
                            for slider in chat_params.values():
                                slider.render()
                            chat_auto_manage_context_checkbox.render()
                            chat_context_status_textbox.render()

                    with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                        chat_rag_form["rag_enabled"].render()
                        chat_rag_form["rag_status"].render()
                        with gr.Row():
                            chat_rag_form["file_upload"].render()
                        with gr.Row():
                            chat_rag_form["upload_button"].render()
                            chat_rag_form["clear_button"].render()
                        chat_rag_form["upload_status"].render()
                        with gr.Group():
                            with gr.Row():
                                rag_params["chunk_size"].render()
                                rag_params["chunk_overlap"].render()
                            with gr.Row():
                                rag_params["n_results"].render()
                                rag_params["similarity_threshold"].render()
                        chat_rag_form["update_params_button"].render()
                        chat_rag_form["params_status"].render()

                with gr.Column(scale=8):
                    with gr.Row(equal_height=True):
                        chat_system_prompt_textbox.render()
                        chat_default_system_prompt_button.render()

                    chatbot = gr.Chatbot(
                        buttons=["copy", "copy_all"],
                        render=False,
                        latex_delimiters=[
                            {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                            {"left": "\\begin{equation*}", "right": "\\end{equation*}", "display": True},
                            {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                            {"left": "\\begin{align*}", "right": "\\end{align*}", "display": True},
                            {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                            {"left": "\\begin{alignat*}", "right": "\\end{alignat*}", "display": True},
                            {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                            {"left": "\\begin{gather*}", "right": "\\end{gather*}", "display": True},
                            {"left": "\\begin{eqnarray}", "right": "\\end{eqnarray}", "display": True},
                            {"left": "\\begin{eqnarray*}", "right": "\\end{eqnarray*}", "display": True},
                            {"left": "\\begin{multline}", "right": "\\end{multline}", "display": True},
                            {"left": "\\begin{multline*}", "right": "\\end{multline*}", "display": True},
                            {"left": "\\begin{split}", "right": "\\end{split}", "display": True},
                            {"left": "\\begin{cases}", "right": "\\end{cases}", "display": True},
                            {"left": "\\begin{matrix}", "right": "\\end{matrix}", "display": True},
                            {"left": "\\begin{pmatrix}", "right": "\\end{pmatrix}", "display": True},
                            {"left": "\\begin{bmatrix}", "right": "\\end{bmatrix}", "display": True},
                            {"left": "\\begin{vmatrix}", "right": "\\end{vmatrix}", "display": True},
                            {"left": "\\begin{Vmatrix}", "right": "\\end{Vmatrix}", "display": True},
                            {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                            {"left": "\\[", "right": "\\]", "display": True},
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "\\(", "right": "\\)", "display": False},
                            {"left": "$", "right": "$", "display": False},
                        ],
                    )
                    chat_textbox = gr.MultimodalTextbox(
                        show_label=False,
                        label="",
                        placeholder=I18nData("chat_interface.message_placeholder"),
                        scale=7,
                        autofocus=False,
                        file_count="multiple",
                        # Gradio 6 ignores ChatInterface stop_btn when a custom textbox is provided.
                        stop_btn=True,
                    )
                    gr.ChatInterface(
                        multimodal=True,
                        chatbot=chatbot,
                        textbox=chat_textbox,
                        editable=True,
                        fn=chat_fn,
                        title=None,
                        autofocus=False,
                        fill_height=True,
                        fill_width=True,
                        save_history=True,
                        additional_inputs=[
                            chat_system_prompt_textbox,
                            *chat_params.values(),
                            chat_rag_form["rag_enabled"],
                            rag_params["n_results"],
                            chat_auto_manage_context_checkbox,
                            chat_context_summary_state,
                            chat_prompt_cache_state,
                        ],
                        additional_outputs=[chat_context_summary_state, chat_context_status_textbox, chat_prompt_cache_state],
                    )

        with gr.Tab(get_text("Tab.completion"), interactive=True):
            with gr.Row():
                with gr.Column(scale=2):
                    completion_memory.render()
                    gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")
                    completion_model_selector.render()
                    completion_model_status.render()
                    completion_load_button.render()
                    with gr.Row(visible=False):
                        with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=True):
                            for slider in completion_params.values():
                                slider.render()

                with gr.Column(scale=8):
                    completion_interface = gr.Interface(
                        clear_btn=None,
                        flagging_mode="never",
                        fn=completion_fn,
                        inputs=[
                            gr.Textbox(
                                lines=10,
                                buttons=["copy"],
                                render=True,
                                label=get_text("Page.Completion.Textbox.prompt.label"),
                            ),
                            *completion_params.values(),
                        ],
                        outputs=[
                            gr.Textbox(
                                lines=25,
                                buttons=["copy"],
                                render=True,
                                label=get_text("Page.Completion.Textbox.output.label"),
                            )
                        ],
                        submit_btn=get_text("Page.Completion.Button.submit.value"),
                        stop_btn=get_text("Page.Completion.Button.stop.value"),
                    )

        with gr.Tab(get_text("Tab.model_management"), interactive=True):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5):
                    model_list.render()
                with gr.Column(scale=5):
                    gr.Markdown(f"## {get_text('Page.ModelManagement.DeleteModelBlock.Markdown.add_model')}")
                    local_model_form["search_query"].render()
                    local_model_form["search_button"].render()
                    local_model_form["search_results"].render()
                    local_model_form["model_name"].render()
                    local_model_form["mlx_repo"].render()
                    local_model_form["quantize"].render()
                    local_model_form["default_language"].render()
                    local_model_form["system_prompt"].render()
                    local_model_form["multimodal_mode"].render()
                    local_model_form["multimodal_ability_override"].render()
                    local_model_form["detected_capabilities"].render()
                    local_model_form["add_button"].render()

                    gr.Markdown(f"## {get_text('Page.ModelManagement.DeleteModelBlock.Markdown.delete_model')}")
                    local_model_form["delete_model_selector"].render()
                    local_model_form["delete_files_checkbox"].render()
                    local_model_form["delete_button"].render()
                    local_model_form["delete_status"].render()

        ui = AppUI(
        app=app,
        timer=timer,
        chat=ChatUI(
            memory=chat_memory,
            model_selector=chat_model_selector,
            model_status=chat_model_status,
            load_button=chat_load_button,
            params=chat_params,
            rag_params=rag_params,
            system_prompt=chat_system_prompt_textbox,
            default_system_prompt_button=chat_default_system_prompt_button,
            auto_manage_context=chat_auto_manage_context_checkbox,
            context_status=chat_context_status_textbox,
            context_summary_state=chat_context_summary_state,
            prompt_cache_state=chat_prompt_cache_state,
            rag_form=chat_rag_form,
            chatbot=chatbot,
        ),
        completion=CompletionUI(
            memory=completion_memory,
            model_selector=completion_model_selector,
            model_status=completion_model_status,
            load_button=completion_load_button,
            params=completion_params,
            interface=completion_interface,
        ),
        model_management=ModelManagementUI(form=local_model_form, model_list=model_list),
        )
        if bind_fn is not None:
            bind_fn(ui)
    return ui
