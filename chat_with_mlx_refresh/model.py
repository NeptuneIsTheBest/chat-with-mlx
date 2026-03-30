from __future__ import annotations

import enum
import gc
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Union

import mlx
import mlx_vlm
from mlx_lm import generate, load, sample_utils, stream_generate
from mlx_lm.generate import GenerationResponse

from .model_config import ModelConfig, ModelConfigStore


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role.value, "content": self.content}


def get_model_max_length(model, tokenizer=None, config=None) -> int:
    search_paths = []
    if config:
        search_paths.extend(
            [
                (config, "text_config.max_position_embeddings"),
                (config, "max_position_embeddings"),
                (config, "sliding_window"),
                (config, "max_sequence_length"),
            ]
        )
    if tokenizer and hasattr(tokenizer, "model_max_length") and tokenizer.model_max_length < 1e6:
        return tokenizer.model_max_length

    search_paths.extend(
        [
            (model, "config.max_position_embeddings"),
            (model, "args.max_position_embeddings"),
            (model, "args.text_config.max_position_embeddings"),
            (model, "args.sliding_window"),
            (model, "text_config.max_position_embeddings"),
        ]
    )

    for obj, path in search_paths:
        current_obj = obj
        try:
            for attr in path.split("."):
                if isinstance(current_obj, dict):
                    current_obj = current_obj.get(attr)
                else:
                    current_obj = getattr(current_obj, attr)
                if current_obj is None:
                    break
            if current_obj is not None:
                return int(current_obj)
        except (AttributeError, TypeError):
            continue

    default_max_len = 32768
    logging.warning("Could not determine model's max length. Falling back to default: %s", default_max_len)
    return default_max_len


class BaseLocalModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.load()

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def format_chat_prompt(self, message: Dict[str, str], history: List[Dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def get_multimodal_abilities(self) -> List[str]:
        return []

    def supports_multimodal_ability(self, ability: str) -> bool:
        return ability in self.get_multimodal_abilities()

    def generate_completion(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        return self.perform_generation(prompt, stream, **kwargs)

    def generate_response(
        self,
        message: str,
        history: List[Dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        user_message = Message(MessageRole.USER, message).to_dict()
        formatted_prompt = self.format_chat_prompt(message=user_message, history=history, **kwargs)
        return self.perform_generation(formatted_prompt, stream, **kwargs)


class TextModel(BaseLocalModel):
    def __init__(self, model_path: str):
        self.model = None
        self.tokenizer = None
        self.max_position_embeddings = None
        super().__init__(model_path)

    def load(self) -> None:
        try:
            self.model, self.tokenizer = load(self.model_path, tokenizer_config={"trust_remote_code": True})
        except Exception as exc:
            raise RuntimeError(f"Failed to load text model {self.model_path}: {exc}") from exc

        try:
            self.max_position_embeddings = get_model_max_length(self.model, self.tokenizer, None)
        except Exception as exc:
            self.max_position_embeddings = None
            logging.warning("Failed to read model max position embeddings %s: %s", self.model_path, exc)

    def format_chat_prompt(self, message: Dict[str, str], history: List[Dict[str, str]], **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=history + [message],
            tokenize=False,
            add_generation_prompt=True,
        )

    def perform_generation(
        self,
        prompt: str,
        stream: bool,
        **kwargs,
    ) -> Union[str, Generator[GenerationResponse, None, None]]:
        gen_params = {
            "temperature": kwargs.get("temperature", 1.0),
            "top_k": kwargs.get("top_k", 20),
            "top_p": kwargs.get("top_p", 0.95),
            "min_p": kwargs.get("min_p", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "presence_penalty": kwargs.get("presence_penalty", 1.5),
        }
        sampler = sample_utils.make_sampler(
            temp=gen_params["temperature"],
            top_k=gen_params["top_k"],
            top_p=gen_params["top_p"],
            min_p=gen_params["min_p"],
        )
        logits_processors = sample_utils.make_logits_processors(
            repetition_penalty=gen_params["repetition_penalty"],
            presence_penalty=gen_params["presence_penalty"],
        )
        gen_args = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "prompt": prompt,
            "sampler": sampler,
            "logits_processors": logits_processors,
            "max_tokens": gen_params["max_tokens"],
        }
        if stream:
            return stream_generate(**gen_args)
        return generate(**gen_args)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        mlx.core.clear_cache()


class MultimodalModel(BaseLocalModel):
    def __init__(self, model_path: str):
        self.config = None
        self.model = None
        self.processor = None
        self.max_position_embeddings = None
        self.multimodal_ability: List[str] = []
        super().__init__(model_path)

    def load(self) -> None:
        try:
            self.config = mlx_vlm.utils.load_config(self.model_path)
            self.model, self.processor = mlx_vlm.load(
                self.model_path,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load multimodal model {self.model_path}: {exc}") from exc

        self.multimodal_ability = ModelConfigStore.detect_multimodal_abilities_from_payload(self.config)

        try:
            self.max_position_embeddings = get_model_max_length(self.model, self.processor, self.config)
        except Exception as exc:
            self.max_position_embeddings = None
            logging.warning("Failed to read model max position embeddings %s: %s", self.model_path, exc)

    def format_chat_prompt(self, message: Dict[str, str], history: List[Dict[str, str]], **kwargs) -> str:
        turn_media = kwargs.get("turn_media", [])
        conversation = history + [message]
        model_type = self.config["model_type"] if isinstance(self.config, dict) else self.config.model_type

        formatted_messages = []
        for index, item in enumerate(conversation):
            media_for_turn = turn_media[index] if index < len(turn_media) else {}
            images_for_turn = media_for_turn.get("images", [])
            audios_for_turn = media_for_turn.get("audios", [])
            is_user_message = item.get("role") == MessageRole.USER.value
            formatted_messages.append(
                mlx_vlm.prompt_utils.get_message_json(
                    model_type,
                    item.get("content", ""),
                    role=item.get("role", MessageRole.USER.value),
                    skip_image_token=not is_user_message or not images_for_turn,
                    skip_audio_token=not is_user_message or not audios_for_turn,
                    num_images=len(images_for_turn) if is_user_message else 0,
                    num_audios=len(audios_for_turn) if is_user_message else 0,
                )
            )

        if model_type in {"paligemma", "molmo", "florence2"}:
            return formatted_messages[-1]

        return mlx_vlm.prompt_utils.get_chat_template(
            self.processor,
            formatted_messages,
            add_generation_prompt=True,
        )

    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        gen_args = {
            "model": self.model,
            "processor": self.processor,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "top_k": kwargs.get("top_k", 20),
            "top_p": kwargs.get("top_p", 0.9),
            "min_p": kwargs.get("min_p", 0.0),
            "max_tokens": kwargs.get("max_tokens", 512),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "image": kwargs.get("images", []),
            "audio": kwargs.get("audios", []),
        }
        if stream:
            return mlx_vlm.stream_generate(**gen_args)
        return mlx_vlm.generate(**gen_args).text

    def get_multimodal_abilities(self) -> List[str]:
        return list(self.multimodal_ability)

    def close(self) -> None:
        del self.model
        del self.processor
        del self.config
        gc.collect()
        mlx.core.clear_cache()


class ModelType(enum.Enum):
    LOCAL = "local"


class MemoryUsageLevel(enum.Enum):
    HIGH = "high"
    STRICT = "strict"
    NONE = "none"


class ModelManager:
    STRICT_MEMORY_RATIO = 0.7
    STRICT_CACHE_RATIO = 0.3
    HIGH_WIRED_MULTIPLIER = 0.5
    HIGH_MEMORY_RATIO = 0.8
    HIGH_CACHE_RATIO = 0.6

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.config_store = ModelConfigStore(base_dir)
        self.model: Optional[BaseLocalModel] = None
        self.model_config: Optional[ModelConfig] = None
        self.model_configs: Dict[str, ModelConfig] = self.config_store.load_configs()
        self.memory_usage_level = MemoryUsageLevel.STRICT
        self.active_generator: List[Any] = []
        self._model_lock = threading.RLock()
        self._active_generator_lock = threading.Lock()
        self.set_memory_usage_level(self.memory_usage_level)

    def refresh_model_configs(self) -> Dict[str, ModelConfig]:
        with self._model_lock:
            self.model_configs = self.config_store.load_configs()
            return dict(self.model_configs)

    def list_model_names(self) -> List[str]:
        with self._model_lock:
            self.model_configs = self.config_store.load_configs()
            return sorted(self.model_configs.keys())

    def load_model(self, model_name: str) -> None:
        with self._model_lock:
            if self.model:
                self.close_model()

            self.model_configs = self.config_store.load_configs()
            model_config = self.model_configs.get(model_name)
            if not model_config:
                raise RuntimeError(f"Model '{model_name}' not found")

            try:
                self._load_local_model(model_config)
                self.model_config = model_config
                logging.info("Successfully loaded model: %s", model_name)
            except Exception as exc:
                self.close_model()
                logging.error("Error loading model '%s': %s", model_name, exc)
                raise RuntimeError(f"Error loading model '{model_name}': {exc}") from exc

    def _load_local_model(self, model_config: ModelConfig) -> None:
        local_model_path = self.config_store.get_model_path(model_config)
        if not local_model_path.exists():
            self._download_model(model_config, local_model_path)

        if model_config.multimodal_ability:
            self.model = MultimodalModel(str(local_model_path))
        else:
            self.model = TextModel(str(local_model_path))

    def _download_model(self, model_config: ModelConfig, local_model_path) -> None:
        logging.info("Downloading model from %s...", model_config.mlx_repo)
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_config.mlx_repo, local_dir=str(local_model_path))
            logging.info("Model downloaded successfully to %s", local_model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to download model from '{model_config.mlx_repo}': {exc}") from exc

    def close_model(self) -> None:
        with self._model_lock:
            self.close_active_generator()
            if self.model:
                try:
                    self.model.close()
                except Exception as exc:
                    logging.error("Error closing model: %s", exc)
            self.model = None
            self.model_config = None
            gc.collect()
            mlx.core.clear_cache()

    def get_loaded_model(self) -> Optional[BaseLocalModel]:
        with self._model_lock:
            return self.model

    def get_loaded_model_config(self) -> Optional[ModelConfig]:
        with self._model_lock:
            return self.model_config

    def has_loaded_model(self) -> bool:
        with self._model_lock:
            return self.model is not None and self.model_config is not None

    def get_system_prompt(self, default: bool = False) -> Optional[str]:
        with self._model_lock:
            if not self.model_config:
                return None
            if not default and self.model_config.custom_system_prompt is not None:
                return self.model_config.custom_system_prompt
            return self.model_config.system_prompt

    def set_custom_prompt(self, custom_system_prompt: str) -> None:
        with self._model_lock:
            if self.model_config:
                self.model_config.custom_system_prompt = custom_system_prompt

    def get_active_memory(self) -> int:
        return mlx.core.get_active_memory()

    def get_cache_memory(self) -> int:
        return mlx.core.get_cache_memory()

    def get_system_memory_usage(self) -> int:
        total_usage = self.get_active_memory() + self.get_cache_memory()
        return total_usage

    def get_device_info(self) -> Dict[str, Union[str, int]]:
        return mlx.core.device_info()

    def get_memory_usage_level(self) -> MemoryUsageLevel:
        return self.memory_usage_level

    def set_memory_usage_level(self, memory_usage_level: MemoryUsageLevel) -> None:
        self.memory_usage_level = memory_usage_level
        self._apply_memory_policy(memory_usage_level)

    def _apply_memory_policy(self, memory_usage_level: MemoryUsageLevel) -> None:
        device_info = self.get_device_info()
        max_recommended_working_set_size = device_info.get("max_recommended_working_set_size")
        memory_size = device_info.get("memory_size")

        if memory_usage_level == MemoryUsageLevel.STRICT:
            self._set_strict_memory_policy(max_recommended_working_set_size, memory_size)
        elif memory_usage_level == MemoryUsageLevel.HIGH:
            self._set_high_memory_policy(max_recommended_working_set_size, memory_size)
        else:
            self._set_unlimited_memory_policy(memory_size)

    def _set_strict_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = max_recommended_size
        available_memory = memory_size - wired_limit
        mlx.core.set_wired_limit(wired_limit)
        mlx.core.set_memory_limit(int(available_memory * self.STRICT_MEMORY_RATIO))
        mlx.core.set_cache_limit(int(available_memory * self.STRICT_CACHE_RATIO))

    def _set_high_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = int(max_recommended_size * self.HIGH_WIRED_MULTIPLIER)
        available_memory = memory_size - wired_limit
        mlx.core.set_wired_limit(wired_limit)
        mlx.core.set_memory_limit(int(available_memory * self.HIGH_MEMORY_RATIO))
        mlx.core.set_cache_limit(int(available_memory * self.HIGH_CACHE_RATIO))

    def _set_unlimited_memory_policy(self, memory_size: int) -> None:
        mlx.core.set_wired_limit(0)
        mlx.core.set_memory_limit(memory_size)
        mlx.core.set_cache_limit(memory_size)

    def set_active_generator(self, gen) -> None:
        with self._active_generator_lock:
            self.active_generator.append(gen)

    def remove_active_generator(self, gen) -> None:
        with self._active_generator_lock:
            if gen in self.active_generator:
                self.active_generator.remove(gen)

    def close_active_generator(self) -> None:
        with self._active_generator_lock:
            generators = list(self.active_generator)

        if not generators:
            return

        successfully_closed = []
        for generator in generators:
            try:
                generator.close()
                close_error = getattr(generator, "get_close_error", None)
                if callable(close_error):
                    iterator_close_error = close_error()
                    if iterator_close_error is not None:
                        logging.warning("Generator %s closed with iterator error: %s", generator, iterator_close_error)
                successfully_closed.append(generator)
            except Exception as exc:
                logging.warning("Failed to close generator %s: %s", generator, exc)

        if successfully_closed:
            with self._active_generator_lock:
                self.active_generator = [
                    generator for generator in self.active_generator if generator not in successfully_closed
                ]

        gc.collect()
        mlx.core.clear_cache()
