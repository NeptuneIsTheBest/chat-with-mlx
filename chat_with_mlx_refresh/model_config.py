from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from huggingface_hub import hf_hub_download


@dataclass
class ModelConfig:
    mlx_repo: str
    model_name: str
    quantize: Optional[str] = None
    default_language: str = "multi"
    system_prompt: Optional[str] = None
    multimodal_ability: list[str] = field(default_factory=list)
    kv_cache_backend: str = "default"
    turboquant_bits: Optional[int] = None
    display_name: Optional[str] = None
    custom_system_prompt: Optional[str] = field(default=None, repr=False, compare=False)
    config_path: Optional[Path] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelConfig":
        return cls(
            mlx_repo=(payload.get("mlx_repo") or "").strip(),
            model_name=(payload.get("model_name") or "").strip(),
            quantize=payload.get("quantize"),
            default_language=payload.get("default_language") or "multi",
            system_prompt=payload.get("system_prompt"),
            multimodal_ability=list(payload.get("multimodal_ability") or []),
            kv_cache_backend=str(payload.get("kv_cache_backend") or "default"),
            turboquant_bits=payload.get("turboquant_bits"),
            display_name=payload.get("display_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mlx_repo": self.mlx_repo,
            "model_name": self.model_name,
            "quantize": self.quantize,
            "default_language": self.default_language,
            "system_prompt": self.system_prompt,
            "multimodal_ability": list(self.multimodal_ability),
            "kv_cache_backend": self.kv_cache_backend,
            "turboquant_bits": self.turboquant_bits,
            "display_name": self.display_name,
        }

    @staticmethod
    def build_display_name(
        model_name: str,
        default_language: str,
        quantize: Optional[str],
        multimodal_ability: Optional[list[str]] = None,
        kv_cache_backend: str = "default",
        turboquant_bits: Optional[int] = None,
    ) -> str:
        parts = [default_language, quantize or "None"]
        if (kv_cache_backend or "").strip().lower() == "turboquant":
            bits = int(turboquant_bits) if turboquant_bits in {3, 4} else 4
            parts.append(f"tq{bits}")
        if multimodal_ability:
            parts.append("+".join(multimodal_ability))
        return f"{model_name}({','.join(parts)})"

    @property
    def resolved_display_name(self) -> str:
        if self.display_name:
            return self.display_name
        return self.build_display_name(
            model_name=self.model_name,
            default_language=self.default_language,
            quantize=self.quantize,
            multimodal_ability=self.multimodal_ability,
            kv_cache_backend=self.kv_cache_backend,
            turboquant_bits=self.turboquant_bits,
        )


class ModelConfigStore:
    CAPABILITY_MODE_TEXT_ONLY = "Text only"
    CAPABILITY_MODE_AUTO_DETECT = "Auto detect"
    CAPABILITY_MODE_MANUAL_OVERRIDE = "Manual override"
    KV_CACHE_BACKEND_DEFAULT = "default"
    KV_CACHE_BACKEND_TURBOQUANT = "turboquant"
    VALID_CAPABILITY_MODES = frozenset(
        {
            CAPABILITY_MODE_TEXT_ONLY,
            CAPABILITY_MODE_AUTO_DETECT,
            CAPABILITY_MODE_MANUAL_OVERRIDE,
        }
    )
    VALID_KV_CACHE_BACKENDS = frozenset({KV_CACHE_BACKEND_DEFAULT, KV_CACHE_BACKEND_TURBOQUANT})
    VALID_QUANTIZE_TYPES = frozenset(
        {"None", "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "bf16", "bf32"}
    )
    VALID_TURBOQUANT_BITS = frozenset({3, 4})
    VALID_LANGUAGES = frozenset({"multi"})
    VALID_MULTIMODAL_ABILITIES = frozenset({"vision", "audio"})
    MULTIMODAL_ABILITY_ORDER = ("vision", "audio")
    VISION_CONFIG_KEYS = frozenset(
        {"vision_config", "image_token_id", "image_token_index", "vision_start_token_id", "vision_end_token_id"}
    )
    AUDIO_CONFIG_KEYS = frozenset(
        {"audio_config", "audio_token_id", "audio_token_index", "audio_start_token_id", "audio_end_token_id"}
    )
    REMOTE_CONFIG_FILENAME = "config.json"
    CONFIG_EXTENSION = ".json"
    SAFE_KEY_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent / "models"
        self.configs_dir = self.base_dir / "configs"
        self.models_dir = self.base_dir / "models"
        self._setup_directories()

    def _setup_directories(self) -> None:
        directories = [self.base_dir, self.configs_dir, self.models_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        missing_dirs = [directory for directory in directories if not directory.is_dir()]
        if missing_dirs:
            raise IOError(f"Failed to create directories: {missing_dirs}")

    @staticmethod
    def _validate_repo_format(repo: str, name: str) -> None:
        repo = repo.strip()
        if len(repo.split("/")) != 2 or not all(repo.split("/")):
            raise ValueError(f"'{name}' must be in 'owner/repo' format, got: '{repo}'")

    def _validate_quantize(self, quantize: str) -> None:
        if quantize not in self.VALID_QUANTIZE_TYPES:
            raise ValueError(f"quantize must be one of {self.VALID_QUANTIZE_TYPES}, got: '{quantize}'")

    def _validate_language(self, default_language: str) -> None:
        if default_language not in self.VALID_LANGUAGES:
            raise ValueError(
                f"default_language must be one of {self.VALID_LANGUAGES}, got: '{default_language}'"
            )

    def _validate_multimodal_ability(self, abilities: Optional[list[str]]) -> None:
        if not abilities:
            return

        abilities_set = set(abilities)
        invalid_abilities = abilities_set - self.VALID_MULTIMODAL_ABILITIES
        if invalid_abilities:
            raise ValueError(f"Invalid multimodal abilities: {invalid_abilities}")

    def _validate_capability_mode(self, multimodal_mode: str) -> None:
        if multimodal_mode not in self.VALID_CAPABILITY_MODES:
            raise ValueError(
                f"multimodal_mode must be one of {self.VALID_CAPABILITY_MODES}, got: '{multimodal_mode}'"
            )

    def _validate_kv_cache_backend(self, kv_cache_backend: str) -> None:
        if kv_cache_backend not in self.VALID_KV_CACHE_BACKENDS:
            raise ValueError(
                f"kv_cache_backend must be one of {self.VALID_KV_CACHE_BACKENDS}, got: '{kv_cache_backend}'"
            )

    def _validate_turboquant_bits(self, turboquant_bits: int) -> None:
        if turboquant_bits not in self.VALID_TURBOQUANT_BITS:
            raise ValueError(
                f"turboquant_bits must be one of {self.VALID_TURBOQUANT_BITS}, got: '{turboquant_bits}'"
            )

    @staticmethod
    def _extract_repo_name(repo: str) -> str:
        return repo.strip().split("/")[-1]

    @classmethod
    def normalize_kv_cache_backend(cls, value: Optional[str]) -> str:
        normalized = str(value or cls.KV_CACHE_BACKEND_DEFAULT).strip().lower()
        if normalized in cls.VALID_KV_CACHE_BACKENDS:
            return normalized
        return cls.KV_CACHE_BACKEND_DEFAULT

    @classmethod
    def normalize_turboquant_bits(cls, value: Any) -> Optional[int]:
        if value in (None, ""):
            return None
        try:
            normalized = int(value)
        except (TypeError, ValueError):
            return None
        return normalized if normalized in cls.VALID_TURBOQUANT_BITS else None

    def _normalize_kv_cache_config(
        self,
        kv_cache_backend: Optional[str],
        turboquant_bits: Any,
    ) -> tuple[str, Optional[int]]:
        normalized_backend = self.normalize_kv_cache_backend(kv_cache_backend)
        self._validate_kv_cache_backend(normalized_backend)
        normalized_bits = self.normalize_turboquant_bits(turboquant_bits)
        if normalized_backend == self.KV_CACHE_BACKEND_DEFAULT:
            return normalized_backend, None
        if normalized_bits is None:
            normalized_bits = 4
        self._validate_turboquant_bits(normalized_bits)
        return normalized_backend, normalized_bits

    @classmethod
    def _slugify(cls, value: str) -> str:
        normalized = value.strip().replace("/", "__")
        normalized = cls.SAFE_KEY_PATTERN.sub("_", normalized)
        normalized = normalized.strip("._-")
        return normalized or "item"

    @staticmethod
    def _short_hash(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]

    def _build_model_directory_name(self, mlx_repo: str) -> str:
        return self._slugify(mlx_repo)

    def _build_config_filename(self, display_name: str) -> str:
        slug = self._slugify(display_name)
        return f"{slug}-{self._short_hash(display_name)}{self.CONFIG_EXTENSION}"

    def _process_multimodal_abilities(self, abilities: Optional[list[str]]) -> list[str]:
        if not abilities:
            return []

        normalized = []
        abilities_set = set(abilities)
        for ability in self.MULTIMODAL_ABILITY_ORDER:
            if ability in abilities_set:
                normalized.append(ability)
        return normalized

    @classmethod
    def _payload_contains_any_key(cls, payload: Any, keys: Iterable[str]) -> bool:
        if isinstance(payload, dict):
            if any(key in payload for key in keys):
                return True
            return any(cls._payload_contains_any_key(value, keys) for value in payload.values())

        if isinstance(payload, list):
            return any(cls._payload_contains_any_key(item, keys) for item in payload)

        return False

    @classmethod
    def detect_multimodal_abilities_from_payload(cls, payload: dict[str, Any]) -> list[str]:
        detected = []
        if cls._payload_contains_any_key(payload, cls.VISION_CONFIG_KEYS):
            detected.append("vision")
        if cls._payload_contains_any_key(payload, cls.AUDIO_CONFIG_KEYS):
            detected.append("audio")
        return detected

    @staticmethod
    def format_multimodal_abilities(abilities: Optional[list[str]]) -> str:
        normalized = abilities or []
        if not normalized:
            return "Text only"
        return " + ".join(ability.capitalize() for ability in normalized)

    def _load_detection_payload(self, config_path: Path) -> dict[str, Any]:
        try:
            with config_path.open("r", encoding="utf-8") as file_obj:
                return json.load(file_obj)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Config not found: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse config file: {config_path}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to read config file: {config_path}") from exc

    def detect_multimodal_abilities_from_model_path(self, model_path: Path) -> list[str]:
        payload = self._load_detection_payload(model_path / self.REMOTE_CONFIG_FILENAME)
        return self.detect_multimodal_abilities_from_payload(payload)

    def detect_multimodal_abilities_from_repo(self, mlx_repo: str) -> list[str]:
        self._validate_repo_format(mlx_repo, "mlx_repo")
        try:
            config_path = Path(
                hf_hub_download(
                    repo_id=mlx_repo,
                    filename=self.REMOTE_CONFIG_FILENAME,
                    repo_type="model",
                )
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch config for '{mlx_repo}': {exc}") from exc

        payload = self._load_detection_payload(config_path)
        return self.detect_multimodal_abilities_from_payload(payload)

    def detect_multimodal_abilities(self, model_config: ModelConfig) -> list[str]:
        model_path = self.get_model_path(model_config)
        if model_path.exists():
            return self.detect_multimodal_abilities_from_model_path(model_path)
        return self.detect_multimodal_abilities_from_repo(model_config.mlx_repo)

    def _generate_display_name(self, model_config: ModelConfig) -> str:
        return ModelConfig.build_display_name(
            model_name=model_config.model_name,
            default_language=model_config.default_language,
            quantize=model_config.quantize,
            multimodal_ability=model_config.multimodal_ability,
            kv_cache_backend=model_config.kv_cache_backend,
            turboquant_bits=model_config.turboquant_bits,
        )

    def get_config_path(self, model_config: ModelConfig) -> Path:
        if model_config.config_path is not None:
            return model_config.config_path
        return self.configs_dir / self._build_config_filename(model_config.resolved_display_name)

    def get_model_path(self, model_config: ModelConfig) -> Path:
        if not model_config.mlx_repo:
            raise RuntimeError(f"'mlx_repo' not specified for model '{model_config.model_name or 'unknown'}'")
        return self.models_dir / self._build_model_directory_name(model_config.mlx_repo)

    def _save_config_to_file(self, model_config: ModelConfig) -> None:
        config_path = self.get_config_path(model_config)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=config_path.parent,
                prefix=f"{config_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as file_obj:
                temp_path = Path(file_obj.name)
                json.dump(model_config.to_dict(), file_obj, ensure_ascii=False, indent=4)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(temp_path, config_path)
            model_config.config_path = config_path
        except OSError as exc:
            cleanup_error: Optional[OSError] = None
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as cleanup_exc:
                    cleanup_error = cleanup_exc
                    logging.warning(
                        "Failed to remove temporary config file %s after save failure for %s: %s",
                        temp_path,
                        config_path,
                        cleanup_exc,
                        exc_info=cleanup_exc,
                    )
            error_message = f"Failed to save config to {config_path}: {exc}"
            if cleanup_error is not None and temp_path is not None:
                error_message = (
                    f"{error_message}. Temporary file may remain at {temp_path}: {cleanup_error}"
                )
            raise RuntimeError(error_message) from exc

    def add_config(
        self,
        mlx_repo: str,
        model_name: Optional[str] = None,
        quantize: str = "None",
        default_language: str = "multi",
        system_prompt: Optional[str] = None,
        multimodal_mode: str = CAPABILITY_MODE_TEXT_ONLY,
        multimodal_ability_override: Optional[list[str]] = None,
        kv_cache_backend: str = KV_CACHE_BACKEND_DEFAULT,
        turboquant_bits: Optional[int] = None,
    ) -> ModelConfig:
        self._validate_repo_format(mlx_repo, "mlx_repo")
        self._validate_quantize(quantize)
        self._validate_language(default_language)
        self._validate_capability_mode(multimodal_mode)
        normalized_kv_cache_backend, normalized_turboquant_bits = self._normalize_kv_cache_config(
            kv_cache_backend,
            turboquant_bits,
        )

        normalized_override = self._process_multimodal_abilities(multimodal_ability_override)
        self._validate_multimodal_ability(normalized_override)

        resolved_multimodal_ability: list[str] = []
        if multimodal_mode == self.CAPABILITY_MODE_TEXT_ONLY:
            resolved_multimodal_ability = []
        elif multimodal_mode == self.CAPABILITY_MODE_AUTO_DETECT:
            try:
                resolved_multimodal_ability = self._process_multimodal_abilities(
                    self.detect_multimodal_abilities_from_repo(mlx_repo)
                )
            except Exception as exc:
                raise RuntimeError(
                    "Auto detect failed for this repository. "
                    "Switch to Text only or Manual override if the model is not mlx-vlm compatible."
                ) from exc
        elif multimodal_mode == self.CAPABILITY_MODE_MANUAL_OVERRIDE:
            if not normalized_override:
                raise ValueError("Select at least one ability when using Manual override.")
            resolved_multimodal_ability = normalized_override

        if resolved_multimodal_ability and normalized_kv_cache_backend == self.KV_CACHE_BACKEND_TURBOQUANT:
            raise ValueError("TurboQuant is currently only available for text-only models.")

        config = ModelConfig(
            mlx_repo=mlx_repo.strip(),
            model_name=(model_name.strip() if model_name else self._extract_repo_name(mlx_repo)),
            quantize=None if quantize == "None" else quantize,
            default_language=default_language,
            system_prompt=system_prompt.strip() if system_prompt else None,
            multimodal_ability=resolved_multimodal_ability,
            kv_cache_backend=normalized_kv_cache_backend,
            turboquant_bits=normalized_turboquant_bits,
        )
        config.display_name = self._generate_display_name(config)
        if config.resolved_display_name in self.load_configs():
            raise ValueError(f"Model '{config.resolved_display_name}' already exists")
        self._save_config_to_file(config)
        return config

    def delete_config(self, model_config: ModelConfig, delete_model_files: bool = False) -> None:
        config_path = self.get_config_path(model_config)
        try:
            if config_path.exists():
                config_path.unlink()
                logging.info("Deleted config file: %s", config_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to delete config file {config_path}: {exc}") from exc

        if not delete_model_files:
            return

        model_path = self.get_model_path(model_config)
        try:
            if model_path.exists():
                shutil.rmtree(model_path)
                logging.info("Deleted model files: %s", model_path)
        except OSError as exc:
            logging.warning("Failed to delete model files %s: %s", model_path, exc)

    def _load_config_file(self, config_file: Path) -> Optional[ModelConfig]:
        try:
            with config_file.open("r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except (json.JSONDecodeError, OSError) as exc:
            logging.error("Error loading config from %s: %s", config_file, exc)
            return None

        model_config = ModelConfig.from_dict(payload)
        model_config.config_path = config_file
        return self._process_config(model_config)

    def _process_config(self, model_config: ModelConfig) -> Optional[ModelConfig]:
        if not model_config.model_name or not model_config.default_language:
            return None

        self._validate_multimodal_ability(model_config.multimodal_ability)
        model_config.multimodal_ability = self._process_multimodal_abilities(model_config.multimodal_ability)
        (
            model_config.kv_cache_backend,
            model_config.turboquant_bits,
        ) = self._normalize_kv_cache_config(model_config.kv_cache_backend, model_config.turboquant_bits)
        model_config.display_name = self._generate_display_name(model_config)
        return model_config

    def load_configs(self) -> dict[str, ModelConfig]:
        model_configs: dict[str, ModelConfig] = {}
        for config_file in self.configs_dir.glob(f"*{self.CONFIG_EXTENSION}"):
            if not config_file.is_file():
                continue

            model_config = self._load_config_file(config_file)
            if model_config is None:
                logging.info("Skipping incomplete config: %s", config_file)
                continue

            if model_config.resolved_display_name in model_configs:
                logging.error("Duplicate model display name detected: %s", model_config.resolved_display_name)
                continue
            model_configs[model_config.resolved_display_name] = model_config
        return model_configs
