LANGUAGE = "en"

MULTI_LANGUAGE = {
    "en": {
        "Tab": {
            "chat": "Chat",
            "completion": "Completion",
            "model_management": "Model Management",
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
                        "value": "Submit"
                    },
                    "stop": {
                        "value": "Stop"
                    }
                },
                "Textbox": {
                    "prompt": {
                        "label": "Prompt"
                    },
                    "output": {
                        "label": "Output"
                    }
                }
            },
            "ModelManagement": {
                "Dataframe": {
                    "model_list": {
                        "headers": "Models"
                    }
                },
                "AddModelBlock": {
                    "Textbox": {
                        "model_name": {
                            "label": "Model name",
                            "placeholder": "If empty, it will be set to the repository name of MLX Community."
                        },
                        "original_repo": {
                            "label": "Original Repository",
                            "placeholder": "The original repository. It should look like microsoft/Phi-3.5-vision-instruct."
                        },
                        "mlx_repo": {
                            "label": "MLX Community Repository",
                            "placeholder": "The MLX community Repository. It should look like mlx-community/Phi-3.5-vision-instruct-8bit."
                        },
                        "default_system_prompt": {
                            "label": "Default System Prompt",
                        }
                    },
                    "Dropdown": {
                        "quantize": {
                            "label": "Quantize"
                        },
                        "default_language": {
                            "label": "Default Language"
                        },
                        "multimodal_ability": {
                            "label": "Multimodal Ability"
                        }
                    },
                    "Button": {
                        "add": {
                            "value": "Add model"
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