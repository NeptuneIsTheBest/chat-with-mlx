from __future__ import annotations

import enum
import functools
import gc
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, Optional, Union

import mlx
import mlx_vlm
from mlx_lm import generate, load, sample_utils, stream_generate
from mlx_lm.generate import GenerationResponse, maybe_quantize_kv_cache
from mlx_lm.models import cache as mlx_lm_cache
from mlx_vlm.generate import DEFAULT_PREFILL_STEP_SIZE, generation_stream as vlm_generation_stream, wired_limit as vlm_wired_limit
from mlx_vlm.models import cache as mlx_vlm_cache

from .model_config import ModelConfig, ModelConfigStore


logger = logging.getLogger(__name__)


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: MessageRole
    content: str

    def to_dict(self) -> dict[str, str]:
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
    def format_chat_prompt(self, message: dict[str, str], history: list[dict[str, str]], **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def get_multimodal_abilities(self) -> list[str]:
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
        history: list[dict[str, str]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[str, Generator[str, None, None]]:
        user_message = Message(MessageRole.USER, message).to_dict()
        formatted_prompt = kwargs.pop("formatted_prompt", None)
        if formatted_prompt is None:
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

    def format_chat_prompt(self, message: dict[str, str], history: list[dict[str, str]], **kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            conversation=history + [message],
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_prompt_tokens(self, prompt: str) -> list[int]:
        bos_token = getattr(self.tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not prompt.startswith(bos_token)
        try:
            tokens = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        except TypeError:
            tokens = self.tokenizer.encode(prompt)
        return list(tokens)

    def make_prompt_cache(self):
        return mlx_lm_cache.make_prompt_cache(self.model)

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
        prompt_token_ids = kwargs.get("prompt_token_ids")
        prompt_cache = kwargs.get("prompt_cache")
        cached_prefix_len = max(0, int(kwargs.get("cached_prefix_len") or 0))
        if prompt_token_ids is not None:
            prompt_tokens = list(prompt_token_ids)
            if cached_prefix_len:
                prompt_tokens = prompt_tokens[cached_prefix_len:]
            gen_args["prompt"] = mlx.core.array(prompt_tokens, dtype=mlx.core.uint32)
        if prompt_cache is not None:
            gen_args["prompt_cache"] = prompt_cache
        if stream:
            return stream_generate(**gen_args)
        return generate(**gen_args)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        mlx.core.clear_cache()


class MultimodalModel(BaseLocalModel):
    def __init__(self, model_path: str):
        self.config = None
        self.model = None
        self.processor = None
        self.max_position_embeddings = None
        self.multimodal_ability: list[str] = []
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

    def format_chat_prompt(self, message: dict[str, str], history: list[dict[str, str]], **kwargs) -> str:
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

    def _default_add_special_tokens(self) -> bool:
        return (
            not hasattr(self.processor, "chat_template")
            if self.model.config.model_type in {"gemma3", "gemma3n"}
            else True
        )

    def make_prompt_cache(self):
        return mlx_vlm_cache.make_prompt_cache(self.model.language_model)

    def prepare_prompt_inputs(self, prompt: str, images: list[str], audios: list[str]) -> dict[str, Any]:
        image_token_index = getattr(self.model.config, "image_token_index", None)
        prepared = mlx_vlm.prepare_inputs(
            self.processor,
            images=images,
            audio=audios,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=self._default_add_special_tokens(),
        )
        normalized = dict(prepared)
        if "attention_mask" in normalized and "mask" not in normalized:
            normalized["mask"] = normalized.pop("attention_mask")
        return normalized

    @staticmethod
    def extract_prompt_token_ids(prepared_inputs: dict[str, Any]) -> list[int]:
        input_ids = prepared_inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Prepared multimodal inputs are missing input_ids.")
        return [int(token) for token in input_ids.flatten().tolist()]

    def _stream_generate_with_prepared_inputs(
        self,
        prompt_cache,
        prepared_generation: dict[str, Any],
        cached_input_prefix_len: int,
        cached_token_prefix_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        max_tokens: int,
        repetition_penalty: float,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 5000,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    ) -> Generator[Any, None, None]:
        input_ids = prepared_generation["input_ids"]
        inputs_embeds = prepared_generation["inputs_embeds"]
        step_kwargs = dict(prepared_generation["step_kwargs"])
        total_prompt_tokens = int(prepared_generation["total_prompt_tokens"])

        sampler = sample_utils.make_sampler(
            temp=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        logits_processors = sample_utils.make_logits_processors(
            repetition_penalty=repetition_penalty,
        )
        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=quantized_kv_start,
            kv_group_size=kv_group_size,
            kv_bits=kv_bits,
        )

        prompt_start = time.perf_counter()
        with vlm_wired_limit(self.model, [vlm_generation_stream]):
            detokenizer = self.processor.detokenizer
            detokenizer.reset()
            tokens = mlx.core.array([], dtype=input_ids.dtype)

            def _step(y, inputs_embeds=None):
                nonlocal tokens, step_kwargs

                with mlx.core.stream(vlm_generation_stream):
                    if "decoder_input_ids" in step_kwargs:
                        outputs = self.model.language_model(
                            cache=prompt_cache,
                            **step_kwargs,
                        )
                    else:
                        outputs = self.model.language_model(
                            y,
                            inputs_embeds=inputs_embeds,
                            cache=prompt_cache,
                            **step_kwargs,
                        )

                    logits = outputs.logits[:, -1, :]
                    if len(logits_processors) > 0 and len(y) > 0:
                        tokens = mlx.core.concat([tokens, y.flatten()])
                        for processor in logits_processors:
                            logits = processor(tokens, logits)

                    quantize_cache_fn(prompt_cache)
                    logprobs = logits - mlx.core.logsumexp(logits)
                    sampled = sampler(logprobs)

                    if outputs.cross_attention_states is not None:
                        step_kwargs = {"cross_attention_states": outputs.cross_attention_states}
                    elif outputs.encoder_outputs is not None:
                        step_kwargs = {"encoder_outputs": outputs.encoder_outputs}
                    else:
                        step_kwargs = {}

                    return sampled, logprobs.squeeze(0)

            if cached_input_prefix_len:
                inputs_embeds = inputs_embeds[:, cached_input_prefix_len:]
                input_ids = input_ids[:, cached_token_prefix_len:]
                step_kwargs = self._slice_prefill_step_kwargs(
                    step_kwargs,
                    cached_input_prefix_len,
                    int(prepared_generation["input_length"]),
                )

            with mlx.core.stream(vlm_generation_stream):
                if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
                    while inputs_embeds.shape[1] > 1:
                        n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                        self.model.language_model(
                            inputs=input_ids[:, :n_to_process],
                            inputs_embeds=inputs_embeds[:, :n_to_process],
                            cache=prompt_cache,
                            n_to_process=n_to_process,
                            **step_kwargs,
                        )
                        quantize_cache_fn(prompt_cache)
                        mlx.core.eval([cache.state for cache in prompt_cache])
                        inputs_embeds = inputs_embeds[:, n_to_process:]
                        input_ids = input_ids[:, n_to_process:]
                        mlx.core.clear_cache()

                    input_ids = input_ids[:, -1:]

                y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

            mlx.core.async_eval(y)
            generated_token_count = 0
            token = None
            prompt_tps = 0.0
            generation_start = time.perf_counter()
            for n in range(max_tokens + 1):
                if n == 0:
                    mlx.core.eval(y)
                    prompt_time = max(time.perf_counter() - prompt_start, 1e-9)
                    prompt_tps = total_prompt_tokens / prompt_time
                    generation_start = time.perf_counter()

                if n == max_tokens:
                    generated_token_count = n
                    break

                next_y, next_logprobs = _step(y[None])
                mlx.core.async_eval(next_y)
                generated_token_count = n + 1

                if self.processor.tokenizer.stopping_criteria(y.item()):
                    token = y.item()
                    break

                detokenizer.add_token(y.item())
                token = y.item()
                yield mlx_vlm.GenerationResult(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    prompt_tokens=total_prompt_tokens,
                    generation_tokens=n + 1,
                    total_tokens=total_prompt_tokens + n + 1,
                    prompt_tps=prompt_tps,
                    generation_tps=(n + 1) / max(time.perf_counter() - generation_start, 1e-9),
                    peak_memory=mlx.core.get_peak_memory() / 1e9,
                )
                if n % 256 == 0:
                    mlx.core.clear_cache()
                y, logprobs = next_y, next_logprobs
            else:
                pass

            detokenizer.finalize()
            yield mlx_vlm.GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=generated_token_count,
                total_tokens=total_prompt_tokens + generated_token_count,
                prompt_tps=prompt_tps,
                generation_tps=generated_token_count / max(time.perf_counter() - generation_start, 1e-9),
                peak_memory=mlx.core.get_peak_memory() / 1e9,
            )
            mlx.core.clear_cache()

    def perform_generation(self, prompt: str, stream: bool, **kwargs) -> Union[str, Generator[str, None, None]]:
        prepared_inputs = kwargs.get("prepared_inputs")
        prompt_cache = kwargs.get("prompt_cache")
        cached_prefix_len = max(0, int(kwargs.get("cached_prefix_len") or 0))
        cached_token_prefix_len = max(0, int(kwargs.get("cached_token_prefix_len") or 0))
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
        }
        if prompt_cache is not None:
            gen_args["prompt_cache"] = prompt_cache

        if prepared_inputs is not None:
            if cached_prefix_len:
                prepared_generation = self._prepare_cached_generation_inputs(prepared_inputs)
                if not self._can_reuse_cached_prefix(
                    prepared_generation,
                    cached_input_prefix_len=cached_prefix_len,
                    cached_token_prefix_len=cached_token_prefix_len,
                ):
                    logger.warning(
                        "Skipping multimodal prompt cache reuse because the cached prefix does not align "
                        "with the prepared inputs (cache_len=%s, token_prefix=%s, input_len=%s, token_len=%s).",
                        cached_prefix_len,
                        cached_token_prefix_len,
                        prepared_generation["input_length"],
                        prepared_generation["token_length"],
                    )
                    gen_args["prompt_cache"] = self.make_prompt_cache()
                    gen_args.update(prepared_inputs)
                else:
                    generator = self._stream_generate_with_prepared_inputs(
                        prompt_cache=prompt_cache,
                        prepared_generation=prepared_generation,
                        cached_input_prefix_len=cached_prefix_len,
                        cached_token_prefix_len=cached_token_prefix_len,
                        temperature=gen_args["temperature"],
                        top_k=gen_args["top_k"],
                        top_p=gen_args["top_p"],
                        min_p=gen_args["min_p"],
                        max_tokens=gen_args["max_tokens"],
                        repetition_penalty=gen_args["repetition_penalty"],
                    )
                    if stream:
                        return generator
                    return "".join(chunk.text for chunk in generator)
            else:
                gen_args.update(prepared_inputs)
        else:
            gen_args["image"] = kwargs.get("images", [])
            gen_args["audio"] = kwargs.get("audios", [])

        if stream:
            return mlx_vlm.stream_generate(**gen_args)
        return mlx_vlm.generate(**gen_args).text

    def _prepare_cached_generation_inputs(self, prepared_inputs: dict[str, Any]) -> dict[str, Any]:
        input_ids = prepared_inputs["input_ids"]
        pixel_values = prepared_inputs.get("pixel_values")
        mask = prepared_inputs.get("mask")
        data_kwargs = {
            key: value
            for key, value in prepared_inputs.items()
            if key not in {"input_ids", "pixel_values", "mask", "attention_mask"}
        }
        embedding_output = self.model.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=mask,
            **data_kwargs,
        )
        step_kwargs = {
            key: value
            for key, value in embedding_output.to_dict().items()
            if key != "inputs_embeds" and value is not None
        }
        return {
            "input_ids": input_ids,
            "inputs_embeds": embedding_output.inputs_embeds,
            "step_kwargs": step_kwargs,
            "input_length": int(embedding_output.inputs_embeds.shape[1]),
            "token_length": int(input_ids.shape[1]),
            "total_prompt_tokens": int(input_ids.size),
        }

    @staticmethod
    def _can_reuse_cached_prefix(
        prepared_generation: dict[str, Any],
        *,
        cached_input_prefix_len: int,
        cached_token_prefix_len: int,
    ) -> bool:
        input_length = int(prepared_generation["input_length"])
        token_length = int(prepared_generation["token_length"])
        return (
            cached_input_prefix_len > 0
            and cached_token_prefix_len > 0
            and cached_input_prefix_len < input_length
            and cached_token_prefix_len < token_length
        )

    def _slice_prefill_step_kwargs(
        self,
        step_kwargs: dict[str, Any],
        cached_input_prefix_len: int,
        input_length: int,
    ) -> dict[str, Any]:
        sequence_axes = {
            "attention_mask_4d": (2,),
            "cross_attention_mask": (2,),
            "full_text_row_masked_out_mask": (2,),
            "visual_pos_masks": (1,),
            "per_layer_inputs": (1,),
            "decoder_inputs_embeds": (1,),
            "attention_mask": (1,),
        }
        sliced_kwargs = dict(step_kwargs)
        for key, axes in sequence_axes.items():
            value = sliced_kwargs.get(key)
            if value is None:
                continue
            for axis in axes:
                value = self._slice_tensor_if_matches_length(
                    value,
                    axis=axis,
                    start=cached_input_prefix_len,
                    expected_length=input_length,
                )
            sliced_kwargs[key] = value
        return sliced_kwargs

    @staticmethod
    def _slice_tensor_if_matches_length(
        value: Any,
        *,
        axis: int,
        start: int,
        expected_length: int,
    ) -> Any:
        if value is None or not hasattr(value, "shape"):
            return value
        ndim = len(value.shape)
        normalized_axis = axis if axis >= 0 else ndim + axis
        if normalized_axis < 0 or normalized_axis >= ndim:
            return value
        if int(value.shape[normalized_axis]) != expected_length:
            return value
        slices = [slice(None)] * ndim
        slices[normalized_axis] = slice(start, None)
        return value[tuple(slices)]

    def get_multimodal_abilities(self) -> list[str]:
        return list(self.multimodal_ability)

    def close(self) -> None:
        self.model = None
        self.processor = None
        self.config = None
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
    DOWNLOAD_COMPLETE_MARKER = ".chat-with-mlx-download-complete"
    PARTIAL_DOWNLOAD_SUFFIX = ".partial"

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.config_store = ModelConfigStore(base_dir)
        self.model: Optional[BaseLocalModel] = None
        self.model_config: Optional[ModelConfig] = None
        self.model_configs: dict[str, ModelConfig] = self.config_store.load_configs()
        self.memory_usage_level = MemoryUsageLevel.STRICT
        self.active_generator: list[Any] = []
        self._model_lock = threading.RLock()
        self._model_condition = threading.Condition(self._model_lock)
        # Serialize load/close transitions so reservations can wait through a model switch.
        self._model_transition_lock = threading.RLock()
        self._active_model_users = 0
        self._model_loading = False
        self._model_closing = False
        self._active_generator_lock = threading.Lock()
        self.set_memory_usage_level(self.memory_usage_level)

    @contextmanager
    def reserve_loaded_model(self) -> Iterator[BaseLocalModel]:
        with self._model_condition:
            while self._model_closing or self._model_loading:
                self._model_condition.wait()
            if self.model is None:
                raise RuntimeError("No model loaded.")
            model = self.model
            self._active_model_users += 1

        try:
            yield model
        finally:
            with self._model_condition:
                self._active_model_users = max(0, self._active_model_users - 1)
                if self._active_model_users == 0:
                    self._model_condition.notify_all()

    def refresh_model_configs(self) -> dict[str, ModelConfig]:
        with self._model_lock:
            self.model_configs = self.config_store.load_configs()
            return dict(self.model_configs)

    def list_model_names(self) -> list[str]:
        with self._model_lock:
            self.model_configs = self.config_store.load_configs()
            return sorted(self.model_configs.keys())

    def load_model(self, model_name: str) -> None:
        loaded_model = None
        with self._model_transition_lock:
            with self._model_condition:
                while self._model_loading or self._model_closing:
                    self._model_condition.wait()
                self._model_loading = True

            try:
                if self.model:
                    self.close_model()

                with self._model_lock:
                    self.model_configs = self.config_store.load_configs()
                    model_config = self.model_configs.get(model_name)
                if not model_config:
                    raise RuntimeError(f"Model '{model_name}' not found")

                try:
                    loaded_model = self._load_local_model(model_config)
                    with self._model_condition:
                        self.model = loaded_model
                        self.model_config = model_config
                    logging.info("Successfully loaded model: %s", model_name)
                except Exception as exc:
                    try:
                        if loaded_model is not None:
                            loaded_model.close()
                    except Exception as close_exc:
                        logging.error("Error closing partially loaded model '%s': %s", model_name, close_exc)
                    finally:
                        gc.collect()
                        mlx.core.clear_cache()
                    with self._model_condition:
                        self.model = None
                        self.model_config = None
                    logging.error("Error loading model '%s': %s", model_name, exc)
                    raise RuntimeError(f"Error loading model '{model_name}': {exc}") from exc
            finally:
                with self._model_condition:
                    self._model_loading = False
                    self._model_condition.notify_all()

    def _load_local_model(self, model_config: ModelConfig) -> BaseLocalModel:
        local_model_path = self.config_store.get_model_path(model_config)
        if self._is_model_download_complete(local_model_path):
            return self._instantiate_local_model(model_config, local_model_path)

        if not local_model_path.exists():
            self._download_model(model_config, local_model_path)
            return self._instantiate_local_model(model_config, local_model_path)

        try:
            model = self._instantiate_local_model(model_config, local_model_path)
        except Exception:
            logging.warning(
                "Model directory %s exists without a completion marker and failed to load. "
                "Attempting to resume download from %s.",
                local_model_path,
                model_config.mlx_repo,
            )
            self._resume_model_download(model_config, local_model_path)
            model = self._instantiate_local_model(model_config, local_model_path)

        self._mark_model_download_complete(local_model_path)
        return model

    def _instantiate_local_model(self, model_config: ModelConfig, local_model_path: Path) -> BaseLocalModel:
        if model_config.multimodal_ability:
            try:
                return MultimodalModel(str(local_model_path))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load '{model_config.resolved_display_name}' as a multimodal model: {exc}. "
                    "If this repository is not actually compatible with mlx-vlm, re-add it in Text only mode."
                ) from exc
        return TextModel(str(local_model_path))

    def _download_model(self, model_config: ModelConfig, local_model_path: Path) -> None:
        partial_model_path = self._get_partial_model_path(local_model_path)
        if self._is_model_download_complete(partial_model_path):
            logging.info(
                "Publishing completed partial download from %s to %s",
                partial_model_path,
                local_model_path,
            )
            self._publish_downloaded_model(partial_model_path, local_model_path)
            logging.info("Model downloaded successfully to %s", local_model_path)
            return

        if partial_model_path.exists():
            logging.info("Resuming partial model download from %s into %s", model_config.mlx_repo, partial_model_path)
        else:
            logging.info("Downloading model from %s into %s", model_config.mlx_repo, partial_model_path)

        self._snapshot_download(model_config, partial_model_path)
        self._mark_model_download_complete(partial_model_path)
        self._publish_downloaded_model(partial_model_path, local_model_path)
        logging.info("Model downloaded successfully to %s", local_model_path)

    def _resume_model_download(self, model_config: ModelConfig, local_model_path: Path) -> None:
        logging.info("Resuming model download from %s into %s", model_config.mlx_repo, local_model_path)
        self._snapshot_download(model_config, local_model_path)

    def _snapshot_download(self, model_config: ModelConfig, local_model_path: Path) -> None:
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo_id=model_config.mlx_repo, local_dir=str(local_model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download model from '{model_config.mlx_repo}' into '{local_model_path}': {exc}"
            ) from exc

    def _publish_downloaded_model(self, partial_model_path: Path, local_model_path: Path) -> None:
        if local_model_path.exists():
            raise RuntimeError(
                f"Cannot publish downloaded model from '{partial_model_path}' because '{local_model_path}' already exists"
            )
        partial_model_path.rename(local_model_path)

    def _mark_model_download_complete(self, model_path: Path) -> None:
        marker_path = self._get_download_complete_marker_path(model_path)
        marker_path.write_text("complete\n", encoding="utf-8")
        logging.info("Marked model download as complete at %s", marker_path)

    def _is_model_download_complete(self, model_path: Path) -> bool:
        return model_path.is_dir() and self._get_download_complete_marker_path(model_path).is_file()

    def _get_partial_model_path(self, local_model_path: Path) -> Path:
        return local_model_path.parent / f"{local_model_path.name}{self.PARTIAL_DOWNLOAD_SUFFIX}"

    def _get_download_complete_marker_path(self, model_path: Path) -> Path:
        return model_path / self.DOWNLOAD_COMPLETE_MARKER

    @staticmethod
    def _append_cleanup_warning(cleanup_warnings: list[str], action: str) -> None:
        if action not in cleanup_warnings:
            cleanup_warnings.append(action)

    def _record_cleanup_warning(self, cleanup_warnings: list[str], action: str, exc: BaseException) -> None:
        logging.warning("Non-fatal cleanup error while attempting to %s: %s", action, exc, exc_info=exc)
        self._append_cleanup_warning(cleanup_warnings, action)

    def _run_cleanup_step(
        self,
        cleanup_warnings: list[str],
        action: str,
        cleanup_fn: Callable[[], Any],
    ) -> None:
        try:
            cleanup_fn()
        except Exception as exc:
            self._record_cleanup_warning(cleanup_warnings, action, exc)

    def close_model(self) -> list[str]:
        model = None
        cleanup_warnings: list[str] = []
        with self._model_transition_lock:
            with self._model_condition:
                while self._model_closing:
                    self._model_condition.wait()
                self._model_closing = True

            try:
                cleanup_warnings.extend(self.close_active_generator(clear_runtime_cache=False))

                with self._model_condition:
                    while self._active_model_users > 0:
                        self._model_condition.wait()

                    model = self.model
                    self.model = None
                    self.model_config = None
            finally:
                with self._model_condition:
                    self._model_closing = False
                    self._model_condition.notify_all()

        if model:
            self._run_cleanup_step(cleanup_warnings, "close the active model", model.close)
        self._run_cleanup_step(cleanup_warnings, "run garbage collection after closing the model", gc.collect)
        self._run_cleanup_step(cleanup_warnings, "clear the MLX cache after closing the model", mlx.core.clear_cache)
        return cleanup_warnings

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

    def get_device_info(self) -> dict[str, Union[str, int]]:
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

        if not isinstance(memory_size, int) or memory_size <= 0:
            logging.warning(
                "Skipping MLX memory policy update for level=%s due to invalid device memory_size=%s. "
                "Keeping existing MLX limits. device_info=%s",
                memory_usage_level.value,
                memory_size,
                device_info,
            )
            return

        if not isinstance(max_recommended_working_set_size, int) or max_recommended_working_set_size < 0:
            logging.warning(
                "Invalid MLX recommended working set size for level=%s: %s. "
                "Applying unlimited memory policy with memory_size=%s. device_info=%s",
                memory_usage_level.value,
                max_recommended_working_set_size,
                memory_size,
                device_info,
            )
            self._set_unlimited_memory_policy(memory_size)
            return

        if memory_usage_level == MemoryUsageLevel.STRICT:
            self._set_strict_memory_policy(max_recommended_working_set_size, memory_size)
        elif memory_usage_level == MemoryUsageLevel.HIGH:
            self._set_high_memory_policy(max_recommended_working_set_size, memory_size)
        else:
            self._set_unlimited_memory_policy(memory_size)

    def _set_strict_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = max(0, min(max_recommended_size, memory_size))
        available_memory = max(0, memory_size - wired_limit)
        mlx.core.set_wired_limit(wired_limit)
        mlx.core.set_memory_limit(int(available_memory * self.STRICT_MEMORY_RATIO))
        mlx.core.set_cache_limit(int(available_memory * self.STRICT_CACHE_RATIO))

    def _set_high_memory_policy(self, max_recommended_size: int, memory_size: int) -> None:
        wired_limit = int(max_recommended_size * self.HIGH_WIRED_MULTIPLIER)
        wired_limit = max(0, min(wired_limit, memory_size))
        available_memory = max(0, memory_size - wired_limit)
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

    def close_active_generator(self, *, clear_runtime_cache: bool = True) -> list[str]:
        with self._active_generator_lock:
            generators = list(self.active_generator)

        cleanup_warnings: list[str] = []
        if not generators:
            return cleanup_warnings

        successfully_closed = []
        for generator in generators:
            try:
                generator.close()
                close_error = getattr(generator, "get_close_error", None)
                if callable(close_error):
                    iterator_close_error = close_error()
                    if iterator_close_error is not None:
                        logging.warning("Generator %s closed with iterator error: %s", generator, iterator_close_error)
                        self._append_cleanup_warning(cleanup_warnings, "close an active generator")
                successfully_closed.append(generator)
            except Exception as exc:
                logging.warning("Failed to close generator %s: %s", generator, exc, exc_info=exc)
                self._append_cleanup_warning(cleanup_warnings, "close an active generator")

        if successfully_closed:
            with self._active_generator_lock:
                self.active_generator = [
                    generator for generator in self.active_generator if generator not in successfully_closed
                ]

        if clear_runtime_cache:
            self._run_cleanup_step(
                cleanup_warnings,
                "run garbage collection after closing active generators",
                gc.collect,
            )
            self._run_cleanup_step(
                cleanup_warnings,
                "clear the MLX cache after closing active generators",
                mlx.core.clear_cache,
            )
        return cleanup_warnings
