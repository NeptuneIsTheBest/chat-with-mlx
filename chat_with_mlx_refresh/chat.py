import gradio as gr

from .components_block import ComponentsBlock
from .language import get_text
from .model import ModelManager


class ChatSystemPromptBlock(ComponentsBlock):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.system_prompt_textbox = gr.Textbox(
            label=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.label"),
            placeholder=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.placeholder"),
            value=self.model_manager.get_system_prompt,
            lines=3,
            show_copy_button=True,
            render=False,
            scale=9
        )
        self.default_system_prompt_button = gr.Button(
            value=get_text("Page.Chat.ChatSystemPromptBlock.Button.default_system_prompt.value"),
            render=False,
            scale=1
        )

    def render_all(self):
        self.system_prompt_textbox.render()
        self.default_system_prompt_button.render()

    def get_system_prompt_text(self):
        try:
            return self.model_manager.get_system_prompt()
        except Exception as e:
            gr.Error(str(e))


class SystemStatusBlock(ComponentsBlock):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.device_name_textbox = gr.Textbox(
            label=get_text("Page.Chat.SystemStatusBlock.Textbox.device_name.label"),
            value=lambda: self.model_manager.get_device_info()["device_name"],
            interactive=False,
            render=False
        )
        self.memory_usage_textbox = gr.Textbox(
            label=get_text("Page.Chat.SystemStatusBlock.Textbox.memory_usage.label"),
            interactive=False,
            render=False
        )

    def render_all(self):
        self.device_name_textbox.render()
        self.memory_usage_textbox.render()


class LoadModelBlock(ComponentsBlock):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.model_selector_dropdown = gr.Dropdown(
            label=get_text("Page.Chat.LoadModelBlock.Dropdown.model_selector.label"),
            choices=self.model_manager.get_model_list(),
            render=False,
            interactive=True
        )
        self.model_status_textbox = gr.Textbox(
            value=self.get_load_model_status,
            show_label=False,
            render=False,
            interactive=False
        )
        self.load_model_button = gr.Button(
            value=get_text("Page.Chat.LoadModelBlock.Button.load_model.value"),
            render=False,
            interactive=True
        )

    def render_all(self):
        self.model_selector_dropdown.render()
        self.model_status_textbox.render()
        self.load_model_button.render()

    def update_select_model_dropdown_value(self):
        if self.model_manager.get_loaded_model_config():
            return self.model_manager.get_loaded_model_config().get("display_name")
        else:
            return self.model_manager.get_model_list()[0] if len(self.model_manager.get_model_list()) > 0 else None

    def get_load_model_status(self):
        if self.model_manager.get_loaded_model_config():
            return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value").format(self.model_manager.get_loaded_model_config().get("display_name"))
        else:
            return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")


class AdvancedSettingBlock(ComponentsBlock):
    def __init__(self):
        self.temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=0.6,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.temperature.label"),
            render=False,
            interactive=True
        )
        self.top_k_slider = gr.Slider(
            minimum=0,
            maximum=100,
            value=20,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.top_k.label"),
            render=False,
            interactive=True
        )
        self.top_p_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.95,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.top_p.label"),
            render=False,
            interactive=True
        )
        self.min_p_slider = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.0,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.min_p.label"),
            render=False,
            interactive=True
        )
        self.max_tokens_slider = gr.Slider(
            minimum=1,
            maximum=32768,
            value=4096,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.max_tokens.label"),
            render=False,
            interactive=True
        )
        self.repetition_penalty_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.repetition_penalty.label"),
            render=False,
            interactive=True
        )

    def render_all(self):
        self.temperature_slider.render()
        self.top_k_slider.render()
        self.top_p_slider.render()
        self.min_p_slider.render()
        self.max_tokens_slider.render()
        self.repetition_penalty_slider.render()


class RAGSettingBlock(ComponentsBlock):
    def __init__(self):
        self.not_implemented_markdown = gr.Markdown(
            value=get_text("Page.Chat.Accordion.RAGSetting.Markdown.not_implemented"),
            render=False
        )

    def render_all(self):
        self.not_implemented_markdown.render()
