import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    family: str
    base_model_name: str
    finetuned_prefix: str
    models_local_root: str
    base_model_local_path: str
    remote_root: str
    base_model_remote_path: str


def _models_local_root() -> str:
    return os.environ.get("LLM_MODELS_LOCAL_ROOT", "./Models")


def _default_remote_root() -> str:
    return os.environ.get(
        "LLM_REMOTE_ROOT",
        "/projects/proj-4180_privacy-1128.4.1354/RobustPipeline/Models",
    )


def _build_qwen_config() -> LLMConfig:
    model_family = os.environ.get("QWEN_MODEL_FAMILY", "Qwen2.5")
    model_size = os.environ.get("QWEN_MODEL_SIZE", "7B")
    model_suffix = os.environ.get("QWEN_MODEL_SUFFIX", "Instruct")

    base_model_name = f"{model_family}-{model_size}-{model_suffix}"
    finetuned_prefix = f"{model_family}-{model_size}-finetuned-FT"
    models_root = _models_local_root()
    remote_root = os.environ.get("QWEN_REMOTE_ROOT", _default_remote_root())

    base_model_local_path = os.path.join(models_root, base_model_name)
    base_model_remote_path = os.path.join(remote_root, base_model_name)

    return LLMConfig(
        family="qwen",
        base_model_name=base_model_name,
        finetuned_prefix=finetuned_prefix,
        models_local_root=models_root,
        base_model_local_path=base_model_local_path,
        remote_root=remote_root,
        base_model_remote_path=base_model_remote_path,
    )


def _build_gemma_config() -> LLMConfig:
    model_family = os.environ.get("GEMMA_MODEL_FAMILY", "Gemma2")
    model_size = os.environ.get("GEMMA_MODEL_SIZE", "9B")
    model_suffix = os.environ.get("GEMMA_MODEL_SUFFIX", "Instruct")

    base_model_name = f"{model_family}-{model_size}" + (f"-{model_suffix}" if model_suffix else "")
    finetuned_prefix = f"{model_family}-{model_size}-finetuned-FT"
    models_root = _models_local_root()
    remote_root = os.environ.get("GEMMA_REMOTE_ROOT", _default_remote_root())

    base_model_local_path = os.path.join(models_root, base_model_name)
    base_model_remote_path = os.path.join(remote_root, base_model_name)

    return LLMConfig(
        family="gemma",
        base_model_name=base_model_name,
        finetuned_prefix=finetuned_prefix,
        models_local_root=models_root,
        base_model_local_path=base_model_local_path,
        remote_root=remote_root,
        base_model_remote_path=base_model_remote_path,
    )


_LLM_BUILDERS = {
    "qwen": _build_qwen_config,
    "gemma": _build_gemma_config,
}


def load_llm_config(llm_family: Optional[str] = None) -> LLMConfig:
    """
    Resolve the configuration for a given LLM family.

    Args:
        llm_family: One of {"qwen", "gemma"}. Defaults to env LLM_FAMILY or "qwen".
    """
    key = (llm_family or os.environ.get("LLM_FAMILY", "qwen")).lower()
    if key not in _LLM_BUILDERS:
        raise ValueError(f"Unsupported llm_family '{llm_family}'. Options: {list(_LLM_BUILDERS)}")
    return _LLM_BUILDERS[key]()


# Default config resolved at import time (respects env LLM_FAMILY)
DEFAULT_LLM_CONFIG = load_llm_config()

BASE_MODEL_NAME = DEFAULT_LLM_CONFIG.base_model_name
FINETUNED_PREFIX = DEFAULT_LLM_CONFIG.finetuned_prefix
MODELS_LOCAL_ROOT = DEFAULT_LLM_CONFIG.models_local_root
BASE_MODEL_LOCAL_PATH = DEFAULT_LLM_CONFIG.base_model_local_path
REMOTE_MODEL_ROOT = DEFAULT_LLM_CONFIG.remote_root
BASE_MODEL_REMOTE_PATH = DEFAULT_LLM_CONFIG.base_model_remote_path

# Compatibility alias; points to the active remote root, not just Qwen.
REMOTE_QWEN_ROOT = REMOTE_MODEL_ROOT
