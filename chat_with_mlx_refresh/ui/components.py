from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import gradio as gr

from ..language import get_text


@dataclass
class ChatUI:
    memory: gr.Textbox
    model_selector: gr.Dropdown
    model_status: gr.Textbox
    load_button: gr.Button
    params: dict[str, gr.Slider]
    rag_params: dict[str, gr.Slider]
    system_prompt: gr.Textbox
    default_system_prompt_button: gr.Button
    auto_manage_context: gr.Checkbox
    context_status: gr.Textbox
    context_summary_state: gr.State
    prompt_cache_state: gr.State
    rag_form: dict[str, Any]
    chatbot: gr.Chatbot


@dataclass
class CompletionUI:
    memory: gr.Textbox
    model_selector: gr.Dropdown
    model_status: gr.Textbox
    load_button: gr.Button
    params: dict[str, gr.Slider]
    interface: gr.Interface


@dataclass
class ModelManagementUI:
    form: dict[str, Any]
    model_list: gr.Dataframe


@dataclass
class AppUI:
    app: gr.Blocks
    timer: gr.Timer
    chat: ChatUI
    completion: CompletionUI
    model_management: ModelManagementUI


def create_slider(min_val, max_val, default_val, label_key, **kwargs):
    return gr.Slider(
        minimum=min_val,
        maximum=max_val,
        value=default_val,
        label=get_text(label_key),
        render=False,
        interactive=True,
        **kwargs,
    )


def create_textbox(label_key, placeholder_key: Optional[str] = None, **kwargs):
    params = {
        "label": get_text(label_key),
        "render": False,
        "interactive": True,
        **kwargs,
    }
    if placeholder_key:
        params["placeholder"] = get_text(placeholder_key)
    return gr.Textbox(**params)


def create_model_controls(model_choices, status_value: Callable[[], str]) -> tuple[gr.Textbox, gr.Dropdown, gr.Textbox, gr.Button]:
    memory_usage = gr.Textbox(
        label=get_text("Page.Chat.SystemStatusBlock.Textbox.memory_usage.label"),
        interactive=False,
        render=False,
    )
    model_selector = gr.Dropdown(
        label=get_text("Page.Chat.LoadModelBlock.Dropdown.model_selector.label"),
        choices=model_choices,
        render=False,
        interactive=True,
    )
    model_status = gr.Textbox(value=status_value, show_label=False, render=False, interactive=False)
    load_button = gr.Button(
        value=get_text("Page.Chat.LoadModelBlock.Button.load_model.value"),
        render=False,
        interactive=True,
    )
    return memory_usage, model_selector, model_status, load_button


def create_generation_params() -> dict[str, gr.Slider]:
    slider_configs = {
        "temperature": (0.0, 2.0, 1.0),
        "top_k": (0, 100, 20),
        "top_p": (0.0, 1.0, 0.95),
        "min_p": (0.0, 1.0, 0.0),
        "max_tokens": (1, 32768, 4096),
        "repetition_penalty": (0.0, 2.0, 1.0),
        "presence_penalty": (0.0, 2.0, 1.5),
    }
    return {
        param: create_slider(
            min_val,
            max_val,
            default,
            f"Page.Chat.Accordion.AdvancedSetting.Slider.{param}.label",
        )
        for param, (min_val, max_val, default) in slider_configs.items()
    }


def create_rag_params(rag_parameters) -> dict[str, gr.Slider]:
    chunk_size, chunk_overlap, n_results, similarity_threshold = rag_parameters
    return {
        "chunk_size": gr.Slider(
            minimum=100,
            maximum=2000,
            value=chunk_size,
            step=50,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.chunk_size.label"),
            render=False,
            interactive=True,
        ),
        "chunk_overlap": gr.Slider(
            minimum=0,
            maximum=500,
            value=chunk_overlap,
            step=10,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.chunk_overlap.label"),
            render=False,
            interactive=True,
        ),
        "n_results": gr.Slider(
            minimum=1,
            maximum=10,
            value=n_results,
            step=1,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.n_results.label"),
            render=False,
            interactive=True,
        ),
        "similarity_threshold": gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=similarity_threshold,
            step=0.05,
            label=get_text("Page.Chat.Accordion.RAGSetting.Slider.similarity_threshold.label"),
            render=False,
            interactive=True,
        ),
    }
