import argparse
import atexit
from typing import Dict, List

import gradio as gr
from gradio.components.chatbot import ChatMessage

from chat import ChatSystemPromptBlock, LoadModelBlock, AdvancedSettingBlock, RAGSettingBlock
from language import get_text
from model import Message, MessageRole, ModelManager

chat_model_manager = ModelManager()
completion_model_manager = ModelManager()

chat_system_prompt_block = ChatSystemPromptBlock()
chat_load_model_block = LoadModelBlock(model_manager=chat_model_manager)
chat_advanced_setting_block = AdvancedSettingBlock()
chat_rag_setting_block = RAGSettingBlock()

completion_system_prompt_block = ChatSystemPromptBlock()
completion_load_model_block = LoadModelBlock(model_manager=completion_model_manager)
completion_advanced_setting_block = AdvancedSettingBlock()

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
        model = chat_model_manager.get_loaded_model()
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

completion_output_state = gr.State(value=True)

def handle_completion(prompt: str,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      max_tokens: int = 512,
                      repetition_penalty: float = 1.0,
                      stream: bool = True):
    try:
        model = completion_model_manager.get_loaded_model()

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
            eos_token = [model.tokenizer.eos_token]
            for chunk in model.generate_completion(
                    prompt=prompt,
                    stream=stream,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty):
                if chunk not in eos_token and completion_output_state.value:
                    response += chunk
                    yield response
    except Exception as e:
        raise gr.Error(str(e))


def chat_load_model_callback(model_name: str):
    return chat_load_model_block.load_model(model_name)


def completion_load_model_callback(model_name: str):
    return completion_load_model_block.load_model(model_name)[0]


with gr.Blocks(fill_height=True, fill_width=True) as app:
    gr.HTML("<h1>Chat with MLX</h1>")

    with gr.Tab(get_text("Tab.chat")):
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(f"## {get_text('Page.Chat.Markdown.configuration')}")

                chat_load_model_block.render_all()
                chat_load_model_block.load_model_button.click(
                    fn=chat_load_model_callback,
                    inputs=[chat_load_model_block.model_selector_dropdown],
                    outputs=[
                        chat_load_model_block.model_status_textbox,
                        chat_system_prompt_block.system_prompt_textbox
                    ]
                )

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=False):
                    chat_advanced_setting_block.render_all()

                with gr.Accordion(label=get_text("Page.Chat.Accordion.RAGSetting.label"), open=False):
                    chat_rag_setting_block.render_all()

            with gr.Column(scale=8):
                with gr.Row(equal_height=True):
                    chat_system_prompt_block.render_all()

                chatbot.render()
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
                completion_load_model_block.load_model_button.click(
                    fn=completion_load_model_callback,
                    inputs=[completion_load_model_block.model_selector_dropdown],
                    outputs=[
                        completion_load_model_block.model_status_textbox
                    ]
                )

                with gr.Accordion(label=get_text("Page.Chat.Accordion.AdvancedSetting.label"), open=True):
                    completion_advanced_setting_block.render_all()

            with gr.Column(scale=8):
                completion_textbox = gr.Textbox(lines=25, render=False)
                completion_interface = gr.Interface(
                    clear_btn=None,
                    flagging_mode="never",
                    fn=handle_completion,
                    inputs=[
                        "textbox"
                    ],
                    outputs=[
                        "textbox"
                    ],
                    additional_inputs=[
                        completion_advanced_setting_block.temperature_slider,
                        completion_advanced_setting_block.top_p_slider,
                        completion_advanced_setting_block.max_tokens_slider,
                        completion_advanced_setting_block.repetition_penalty_slider
                    ]
                )

    with gr.Tab(get_text("Tab.model_manager"), interactive=True):
        gr.Markdown("# Not implemented yet.")


def exit_handler():
    if chat_model_manager.model:
        chat_model_manager.model.close()

    if completion_model_manager.model:
        completion_model_manager.model.close()


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
