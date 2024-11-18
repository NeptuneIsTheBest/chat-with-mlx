import argparse
import atexit
from typing import Dict, List, Tuple

import gradio as gr
from gradio.components.chatbot import ChatMessage

from chat import ChatSystemPromptBlock, LoadModelBlock, AdvancedSettingBlock, RAGSettingBlock
from language import get_text
from model import Message, MessageRole, ModelManager, Model

model_manager = ModelManager()


chat_system_prompt_block = ChatSystemPromptBlock(model_manager=model_manager)
chat_load_model_block = LoadModelBlock(model_manager=model_manager)
chat_advanced_setting_block = AdvancedSettingBlock()
chat_rag_setting_block = RAGSettingBlock()

completion_load_model_block = LoadModelBlock(model_manager=model_manager)

completion_advanced_setting_block = AdvancedSettingBlock()


def get_loaded_model() -> Model:
    model = model_manager.get_loaded_model()
    if model is None:
        raise RuntimeError("No model loaded.")
    return model


def handle_chat(message: Dict,
                history: List[Dict],
                system_prompt: str = None,
                temperature: float = 0.7,
                top_p: float = 0.9,
                max_tokens: int = 512,
                repetition_penalty: float = 1.0,
                stream: bool = True):
    try:
        model = get_loaded_model()
        if system_prompt and system_prompt.strip() != "":
            history = [Message(MessageRole.SYSTEM, content=system_prompt).to_dict()] + history

        temperature = float(temperature)
        top_p = float(top_p)
        repetition_penalty = float(repetition_penalty)

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
            eos_token = model.tokenizer.eos_token
            for chunk in model.generate_response(
                    message=message,
                    history=history,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty):
                if eos_token not in chunk:
                    response.content += chunk
                    yield response
    except Exception as e:
        raise gr.Error(str(e))


def handle_completion(prompt: str,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      max_tokens: int = 512,
                      repetition_penalty: float = 1.0,
                      stream: bool = True):
    try:
        model = get_loaded_model()

        temperature = float(temperature)
        top_p = float(top_p)
        repetition_penalty = float(repetition_penalty)

        if not stream:
            return prompt + model.generate_completion(
                prompt=prompt,
                stream=stream,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty)
        else:
            response = prompt
            eos_token = model.tokenizer.eos_token
            for chunk in model.generate_completion(
                    prompt=prompt,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty):
                if eos_token not in chunk:
                    response += chunk
                    yield response
    except Exception as e:
        raise gr.Error(str(e))


def get_load_model_status():
    if model_manager.get_loaded_model_config():
        return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.loaded_value").format(model_manager.get_loaded_model_config().get("display_name"))
    else:
        return get_text("Page.Chat.LoadModelBlock.Textbox.model_status.not_loaded_value")


def load_model(model_name: str) -> Tuple[str, str]:
    try:
        model_manager.load_model(model_name)
        return get_load_model_status(), model_manager.get_system_prompt(default=True)
    except Exception as e:
        gr.Error(str(e))


def chat_load_model_callback(model_name: str):
    return load_model(model_name)


def completion_load_model_callback(model_name: str):
    return load_model(model_name)[0]


def get_default_system_prompt_callback():
    if model_manager.get_loaded_model_config():
        return model_manager.get_system_prompt(default=True)
    else:
        raise gr.Error("No model loaded.")


with gr.Blocks(fill_height=True, fill_width=True, title="Chat with MLX") as app:
    gr.HTML("<h1>Chat with MLX</h1>")

    with gr.Tab(get_text("Tab.chat")):
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Row():
                    gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                    chat_load_model_block.render_all()
                    chat_load_model_block.model_selector_dropdown.select(
                        fn=lambda x: x,
                        inputs=[chat_load_model_block.model_selector_dropdown],
                        outputs=[completion_load_model_block.model_selector_dropdown]
                    )
                    chat_load_model_block.load_model_button.click(
                        fn=chat_load_model_callback,
                        inputs=[chat_load_model_block.model_selector_dropdown],
                        outputs=[
                            chat_load_model_block.model_status_textbox,
                            chat_system_prompt_block.system_prompt_textbox
                        ]
                    ).then(
                        fn=lambda x: x,
                        inputs=[chat_load_model_block.model_status_textbox],
                        outputs=[completion_load_model_block.model_status_textbox]
                    )

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                    chat_advanced_setting_block.render_all()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                    chat_rag_setting_block.render_all()

            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    chat_system_prompt_block.render_all()
                    chat_system_prompt_block.default_system_prompt_button.click(
                        fn=get_default_system_prompt_callback,
                        outputs=[chat_system_prompt_block.system_prompt_textbox]
                    )
                    chat_system_prompt_block.system_prompt_textbox.change(
                        fn=model_manager.set_custom_prompt,
                        inputs=[chat_system_prompt_block.system_prompt_textbox]
                    )
                    
                chatbot = gr.Chatbot(
                    type="messages",
                    show_copy_button=True,
                    render=False,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\(", "right": "\\)", "display": False},
                        {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
                        {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
                        {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
                        {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
                        {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
                        {"left": "\\[", "right": "\\]", "display": True}
                    ]
                )

                gr.ChatInterface(
                    multimodal=True,
                    chatbot=chatbot,
                    type="messages",
                    fn=handle_chat,
                    title=None,
                    autofocus=False,
                    additional_inputs=[
                        chat_system_prompt_block.system_prompt_textbox,
                        chat_advanced_setting_block.temperature_slider,
                        chat_advanced_setting_block.top_p_slider,
                        chat_advanced_setting_block.max_tokens_slider,
                        chat_advanced_setting_block.repetition_penalty_slider
                    ]
                )

    with gr.Tab(get_text("Tab.completion"), interactive=True):
        with gr.Row():
            advanced_setting_accordion = None
            with gr.Column(scale=2):
                gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                completion_load_model_block.render_all()
                completion_load_model_block.model_selector_dropdown.select(
                    fn=lambda x: x,
                    inputs=[completion_load_model_block.model_selector_dropdown],
                    outputs=[chat_load_model_block.model_selector_dropdown]
                )
                completion_load_model_block.load_model_button.click(
                    fn=completion_load_model_callback,
                    inputs=[completion_load_model_block.model_selector_dropdown],
                    outputs=[
                        completion_load_model_block.model_status_textbox
                    ]
                ).then(
                    fn=lambda x: x,
                    inputs=[completion_load_model_block.model_status_textbox],
                    outputs=[chat_load_model_block.model_status_textbox]
                )

                with gr.Row(visible=False):
                    with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=True):
                        completion_advanced_setting_block.render_all()

            with gr.Column(scale=8):
                completion_textbox = gr.Textbox(lines=25, render=False)
                completion_interface = gr.Interface(
                    clear_btn=None,
                    flagging_mode="never",
                    fn=handle_completion,
                    inputs=[
                        gr.Textbox(lines=10, show_copy_button=True, render=True, label=get_text("Page.Completion.Textbox.prompt.label")),
                        completion_advanced_setting_block.temperature_slider,
                        completion_advanced_setting_block.top_p_slider,
                        completion_advanced_setting_block.max_tokens_slider,
                        completion_advanced_setting_block.repetition_penalty_slider
                    ],
                    outputs=[
                        gr.Textbox(lines=25, show_copy_button=True, render=True, label=get_text("Page.Completion.Textbox.output.label"))
                    ],
                    submit_btn=get_text("Page.Completion.Button.submit.value"),
                    stop_btn=get_text("Page.Completion.Button.stop.value"),
                )

    with gr.Tab(get_text("Tab.model_manager"), interactive=True):
        gr.Markdown("# Not implemented yet.")


def exit_handler():
    if model_manager.get_loaded_model():
        model_manager.close_model()


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
