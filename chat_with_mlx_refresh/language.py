LANGUAGE = "en"

MULTI_LANGUAGE = {
    "en": {
        "Tab": {
            "chat": "Chat",
            "completion": "Completion",
            "model_management": "Model Management"
        },
        "Page": {
            "Chat": {
                "Markdown": {
                    "configuration": "Configuration",
                },
                "SystemStatusBlock": {
                    "Textbox": {
                        "memory_usage": {
                            "label": "Memory Usage"
                        }
                    }
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
                            "label": "Select Model"
                        }
                    },
                    "Textbox": {
                        "model_status": {
                            "not_loaded_value": "No model loaded.",
                            "loaded_value": "{} model is loaded."
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
                            "top_k": {
                                "label": "Top K"
                            },
                            "top_p": {
                                "label": "Top P"
                            },
                            "min_p": {
                                "label": "Min P"
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
                        "Button": {
                            "upload": {
                                "value": "Upload File"
                            },
                            "clear": {
                                "value": "Clear Index"
                            },
                            "update_params": {
                                "value": "Update RAG parameters"
                            }
                        },
                        "Checkbox": {
                            "rag_enabled": {
                                "label": "Enable RAG"
                            }
                        },
                        "File": {
                            "file_upload": {
                                "label": "File Upload"
                            }
                        },
                        "Slider": {
                            "chunk_size": {
                                "label": "Chunk Size"
                            },
                            "chunk_overlap": {
                                "label": "Chunk Overlap"
                            },
                            "n_results": {
                                "label": "Number of results"
                            },
                            "similarity_threshold": {
                                "label": "Similarity Threshold"
                            }
                        },
                        "Textbox": {
                            "upload_status": {
                                "label": "Upload Status"
                            },
                            "rag_status": {
                                "label": "RAG Status"
                            },
                            "params_status": {
                                "label": "Parameters update status"
                            }
                        }
                    }
                }
            },
            "Completion": {
                "Markdown": {
                    "configuration": "Configuration"
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
                "Tab": {
                    "local_model": "Local Model",
                    "openai_api": "OpenAI API"
                },
                "Dataframe": {
                    "model_list": {
                        "headers": "Models"
                    }
                },
                "AddLocalModelBlock": {
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
                            "label": "Default System Prompt"
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
                },
                "AddAPIModelBlock": {
                    "Textbox": {
                        "model_name": {
                            "label": "Model name",
                            "placeholder": "The model to use when calling the API."
                        },
                        "nick_name": {
                            "label": "Nick name"
                        },
                        "api_key": {
                            "label": "API Key",
                            "placeholder": "API Secret key."
                        },
                        "base_url": {
                            "label": "Base url",
                            "placeholder": "Base URL."
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
            return path
    return value
