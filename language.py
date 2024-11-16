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
            },
            "Completion": {
                "Markdown": {
                    "configuration": "Configuration",
                },
                "Button": {
                    "submit": {
                        "submit_value": "Submit",
                        "stop_value": "Stop"
                    }
                },
                "Textbox": {
                    "input_output": {
                        "label": "Completion",
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
