import gradio as gr

from .components_block import ComponentsBlock
from .language import get_text
from .model import ModelManager


class AddLocalModelBlock(ComponentsBlock):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.model_name_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.label"),
            placeholder=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.model_name.placeholder"),
            interactive=True
        )
        self.original_repo_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.original_repo.label"),
            placeholder=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.original_repo.placeholder"),
            interactive=True
        )
        self.mlx_repo_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.label"),
            placeholder=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.mlx_repo.placeholder"),
            interactive=True
        )
        self.quantize_dropdown = gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.quantize.label"),
            choices=["None", "4bit", "8bit", "bf16", "bf32"],
            value="None",
            interactive=True
        )
        self.default_language_dropdown = gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.default_language.label"),
            choices=["multi"],
            interactive=True
        )
        self.default_system_prompt_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.default_system_prompt.label"),
            interactive=True
        )
        self.multimodal_ability_dropdown = gr.Dropdown(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Dropdown.multimodal_ability.label"),
            choices=["None", "vision"],
            value="None",
            multiselect=True
        )
        self.add_button = gr.Button(
            value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.add.value")
        )

    def render_all(self):
        self.model_name_textbox.render()
        self.original_repo_textbox.render()
        self.mlx_repo_textbox.render()
        self.quantize_dropdown.render()
        self.default_language_dropdown.render()
        self.default_system_prompt_textbox.render()
        self.multimodal_ability_dropdown.render()
        self.add_button.render()


class AddAPIModelBlock(ComponentsBlock):
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager

        self.model_name_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.model_name.label"),
            placeholder=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.model_name.placeholder"),
            interactive=True
        )
        self.nick_name_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.nick_name.label"),
            interactive=True
        )
        self.api_key_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.api_key.label"),
            placeholder=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.api_key.placeholder"),
            interactive=True
        )
        self.base_url_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.base_url.label"),
            placeholder=get_text("Page.ModelManagement.AddAPIModelBlock.Textbox.base_url.placeholder"),
            interactive=True
        )
        self.default_system_prompt_textbox = gr.Textbox(
            label=get_text("Page.ModelManagement.AddLocalModelBlock.Textbox.default_system_prompt.label"),
            interactive=True
        )
        self.add_button = gr.Button(
            value=get_text("Page.ModelManagement.AddLocalModelBlock.Button.add.value")
        )

    def render_all(self):
        self.model_name_textbox.render()
        self.nick_name_textbox.render()
        self.api_key_textbox.render()
        self.base_url_textbox.render()
        self.default_system_prompt_textbox.render()
        self.add_button.render()
