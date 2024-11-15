import argparse
import atexit
import enum
import json
import os
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import gradio as gr
from gradio.components.chatbot import ChatMessage
from huggingface_hub import snapshot_download

from model import Model, Message, MessageRole

LANGUAGE = "en"

MULTI_LANGUAGE = {
    "en": {
        "Tab": {
            "chat": "Chat",
            "completion": "Completion",
            "model_manager": "Model Manager",
        },
        "Page": {
            "Chat": {
                "Markdown": {
                    "configuration": "Configuration",
                },
                "ChatSystemPromptBlock": {
                    "Textbox": {
                        "system_prompt": {
                            "placeholder": "System prompt. If empty, the model default prompt is used.",
                            "label": "System prompt"
                        }
                    },
                    "Button": {
                        "default_system_prompt": {
                            "value": "Default"
                        }
                    }
                },
                "LoadModelBlock": {
                    "Dropdown": {
                        "model_selector": {
                            "label": "Select Model",
                        }
                    },
                    "Textbox": {
                        "model_status": {
                            "not_loaded_value": "No model loaded.",
                            "loaded_value": "{} model is loaded.",
                        }
                    },
                    "Button": {
                        "load_model": {
                            "value": "Load Model"
                        }
                    }
                },
                "Accordion": {
                    "AdvancedSetting": {
                        "label": "Advanced Setting",
                        "Slider": {
                            "temperature": {
                                "label": "Temperature"
                            },
                            "top_p": {
                                "label": "Top P"
                            },
                            "max_tokens": {
                                "label": "Max Tokens"
                            },
                            "repetition_penalty": {
                                "label": "Repetition Penalty"
                            },
                            "diversity_penalty": {
                                "label": "Diversity Penalty"
                            }
                        }
                    },
                    "RAGSetting": {
                        "label": "RAG Setting",
                        "Markdown": {
                            "not_implemented": "Not implemented yet."
                        }
                    }
                }
            }
        }
    }
}


def get_text(path: str) -> str:
    keys = path.split(".")
    value = MULTI_LANGUAGE[LANGUAGE]
    for key in keys:
        value = value.get(key)
        if value is None:
            return ""
    return value


class LoadModelStatus(enum.Enum):
    NOT_LOADED = get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")
    LOADED = get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value")


class ModelManager:
    def __init__(self, models_folder: Optional[str] = None):
        self.models_folder = Path(models_folder) if models_folder else Path(os.getenv("MODELS_PATH", "./models"))

        self.configs_folder = self.models_folder / "configs"
        self.models_folder_path = self.models_folder / "models"
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.configs_folder.mkdir(parents=True, exist_ok=True)
        self.models_folder_path.mkdir(parents=True, exist_ok=True)

        if not (self.models_folder.is_dir() and self.configs_folder.is_dir() and self.models_folder_path.is_dir()):
            raise IOError("Models folder exists but is not a directory structure as expected.")

        self.model: Optional[Model] = None
        self.model_configs: Dict[str, Dict[str, str]] = self.scan_models()

    def scan_models(self) -> Dict[str, Dict[str, str]]:
        model_configs = {}
        for config_file in self.configs_folder.glob("*.json"):
            if config_file.is_file():
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        model_config = json.load(f)
                    model_name = model_config.get("model_name")
                    default_language = model_config.get("default_language")
                    quantize = model_config.get("quantize")
                    if not all([model_name, default_language, quantize]):
                        print(f"Skipping incomplete config: {config_file}")
                        continue
                    key = f"{model_name}({default_language},{quantize})"
                    model_configs[key] = model_config
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {config_file}: {e}")
        return model_configs

    def load_model(self, model_name: str) -> Tuple[str, Optional[str]]:
        if self.model:
            self.model.close()
            self.model = None

        model_config = self.model_configs.get(model_name)
        if not model_config:
            raise RuntimeError(f"Model '{model_name}' not found.")

        mlx_repo = model_config.get("mlx_repo")
        if not mlx_repo:
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_name}'.")

        mlx_repo = mlx_repo.strip()
        mlx_repo_name = mlx_repo.split("/")[-1]

        local_model_path = self.models_folder_path / mlx_repo_name
        if not local_model_path.exists():
            try:
                local_model_path.mkdir(parents=True, exist_ok=True)
                snapshot_download(repo_id=mlx_repo, local_dir=str(local_model_path))
            except Exception as e:
                raise RuntimeError(f"Failed to download model from '{mlx_repo}': {e}")

        try:
            self.model = Model(str(local_model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{local_model_path}': {e}")

        system_prompt = model_config.get("system_prompt")
        return model_name, system_prompt

    def get_loaded_model(self) -> Model:
        if self.model:
            return self.model
        else:
            raise RuntimeError("No model loaded.")

    def get_model_list(self) -> list:
        return list(self.model_configs.keys())


class ComponentsBlock:
    def render_all(self):
        pass


class ChatSystemPromptBlock(ComponentsBlock):
    def __init__(self):
        self.system_prompt_textbox = gr.Textbox(
            label=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.label"),
            placeholder=get_text("Page.Chat.ChatSystemPromptBlock.Textbox.system_prompt.placeholder"),
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
            value=LoadModelStatus.NOT_LOADED.value,
            show_label=False,
            render=False,
            interactive=False,
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

    def load_model(self, model_name: str):
        try:
            model_name_loaded, system_prompt = self.model_manager.load_model(model_name)
            status_text = LoadModelStatus.LOADED.value.format(model_name_loaded)
            return status_text, system_prompt
        except Exception as e:
            raise gr.Error(str(e))


class AdvancedSettingBlock(ComponentsBlock):
    def __init__(self):
        self.temperature_slider = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.0,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.temperature.label"),
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
            value=1.2,
            label=get_text("Page.Chat.Accordion.AdvancedSetting.Slider.repetition_penalty.label"),
            render=False,
            interactive=True
        )

    def render_all(self):
        self.temperature_slider.render()
        self.top_p_slider.render()
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


model_manager = ModelManager()
chat_system_prompt_block = ChatSystemPromptBlock()
load_model_block = LoadModelBlock(model_manager=model_manager)
advanced_setting_block = AdvancedSettingBlock()
rag_setting_block = RAGSettingBlock()

chatbot = gr.Chatbot(
    type="messages",
    render=False,
    show_copy_button=True,
)


def handle_chat(message: Dict,
                history: List[Dict],
                system_prompt: str = None,
                temperature: float = 0.7,
                top_p: float = 0.9,
                max_tokens: int = 512,
                repetition_penalty: float = 1.0,
                stream: bool = True):
    try:
        model = model_manager.get_loaded_model()
        if system_prompt and system_prompt.strip() != "":
            history = [Message(MessageRole.SYSTEM, content=system_prompt).to_dict()] + history

        if not stream:
            return ChatMessage(role="assistant",
                        content=model.generate_response(
                            message=message,
                            history=history,
                            stream=stream,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            repetition_penalty=repetition_penalty)
                        )
        else:
            response = ChatMessage(role="assistant", content="")
            eos_token = [model.tokenizer.eos_token]
            for chunk in model.generate_response(
                            message=message,
                            history=history,
                            stream=stream,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            repetition_penalty=repetition_penalty):
                if chunk not in eos_token:
                    response.content += chunk
                    yield response
    except Exception as e:
        raise gr.Error(str(e))


with gr.Blocks(fill_height=True, fill_width=True) as app:
    gr.HTML("<h1>Chat with MLX</h1>")

    with gr.Tab(get_text("Tab.chat")):
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                load_model_block.render_all()
                load_model_block.load_model_button.click(
                    fn=load_model_block.load_model,
                    inputs=[load_model_block.model_selector_dropdown],
                    outputs=[
                        load_model_block.model_status_textbox,
                        chat_system_prompt_block.system_prompt_textbox
                    ]
                )

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                    advanced_setting_block.render_all()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                    rag_setting_block.render_all()

            with gr.Column(scale=9):
                with gr.Row(equal_height=True):
                    chat_system_prompt_block.render_all()

                chatbot.render()
                gr.ChatInterface(
                    multimodal=True,
                    chatbot=chatbot,
                    type="messages",
                    fn=handle_chat,
                    title=None,
                    autofocus=True,
                    additional_inputs=[
                        chat_system_prompt_block.system_prompt_textbox,
                        advanced_setting_block.temperature_slider,
                        advanced_setting_block.top_p_slider,
                        advanced_setting_block.max_tokens_slider,
                        advanced_setting_block.repetition_penalty_slider
                    ]
                )

    with gr.Tab(get_text("Tab.completion"), interactive=True):
        gr.Markdown("# Not implemented yet.")

    with gr.Tab(get_text("Tab.model_manager"), interactive=True):
        gr.Markdown("# Not implemented yet.")


def exit_handler():
    if model_manager.model:
        model_manager.model.close()


atexit.register(exit_handler)


def start(port, share=False, in_browser=True):
    app.launch(server_port=port, inbrowser=in_browser, share=share)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chat with MLX"
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    start(args.port, share=args.share)
