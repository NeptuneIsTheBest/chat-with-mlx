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
                "Dataframe": {
                    "model_list": {
                        "headers": "Models"
                    },
                    "search_results": {
                        "headers": ["Model ID", "Likes", "Downloads"]
                    }
                },
                "AddLocalModelBlock": {
                    "Textbox": {
                        "search_query": {
                            "label": "Search HuggingFace (MLX Models)",
                            "placeholder": "e.g. llama"
                        },
                        "model_name": {
                            "label": "Model name",
                            "placeholder": "If empty, it will be set to the repository name of MLX Community."
                        },
                        "mlx_repo": {
                            "label": "MLX Repository",
                            "placeholder": "The MLX repository. It should look like mlx-community/Phi-3.5-vision-instruct-8bit."
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
                        "search": {
                            "value": "Search"
                        },
                        "add": {
                            "value": "Add model"
                        },
                        "delete": {
                            "value": "Delete Model"
                        }
                    }
                },
                "DeleteModelBlock": {
                    "Markdown": {
                        "add_model": "Add Model",
                        "delete_model": "Delete Model"
                    },
                    "Dropdown": {
                        "model_selector": {
                            "label": "Select Model to Delete"
                        }
                    },
                    "Checkbox": {
                        "delete_files": {
                            "label": "Also delete model files (recommended to free disk space)"
                        }
                    },
                    "Textbox": {
                        "delete_status": {
                            "label": "Delete Status"
                        }
                    },
                    "Messages": {
                        "no_model_selected": "Please select a model to delete",
                        "config_deleted": "Model configuration '{}' has been deleted successfully.",
                        "config_and_files_deleted": "Model '{}' and its files have been deleted successfully."
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
