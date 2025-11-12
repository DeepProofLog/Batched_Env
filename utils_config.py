import ast
import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence
from typing import Optional, List
import torch


def get_available_gpus(min_free_gb: float = 1.0) -> List[int]:
    """
    Get list of all GPUs with sufficient free memory.
    
    Args:
        min_free_gb: Minimum free memory in GB required
    
    Returns:
        List of GPU indices with sufficient free memory
    """
    if not torch.cuda.is_available():
        return []
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return []
    
    available_gpus = []
    min_free_bytes = min_free_gb * 1e9
    
    for gpu_id in range(num_gpus):
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(gpu_id)
            used_memory = total_memory - free_memory
            
            print(f"GPU {gpu_id}: {free_memory / 1e9:.2f} GB free / {total_memory / 1e9:.2f} GB total "
                  f"({used_memory / 1e9:.2f} GB used, {100 * used_memory / total_memory:.1f}% utilized)")
            
            if free_memory >= min_free_bytes:
                available_gpus.append(gpu_id)
        except Exception as e:
            print(f"Warning: Could not query GPU {gpu_id}: {e}")
            continue
    
    return available_gpus


BOOLEAN_TRUE = {'true', 't', 'yes', 'y', 'on', '1'}
BOOLEAN_FALSE = {'false', 'f', 'no', 'n', 'off', '0'}


def load_experiment_configs(config_path: str) -> Sequence[Dict[str, Any]]:
    """Load experiments from a YAML file containing an 'experiments' list."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "PyYAML is required to read experiment configuration files. "
            "Install it with `pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(path.read_text())
    if data is None:
        raise ValueError("Configuration file is empty.")

    if not isinstance(data, Mapping) or "experiments" not in data:
        raise ValueError(
            "Configuration file must be a YAML mapping with an 'experiments' list."
        )

    experiments = data["experiments"]
    if not isinstance(experiments, Sequence):
        raise ValueError("The 'experiments' entry must be a list.")

    for idx, experiment in enumerate(experiments):
        if not isinstance(experiment, Mapping):
            raise ValueError(f"Experiment entry at index {idx} must be a dictionary.")

    return list(experiments)


def parse_scalar(
    text: str,
    *,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> Any:
    """Best-effort conversion of a string literal to Python types."""
    text = text.strip()
    if not text:
        return ''
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        lowered = text.lower()
        if lowered in boolean_true:
            return True
        if lowered in boolean_false:
            return False
        if lowered in {'none', 'null'}:
            return None
    return text


def coerce_config_value(
    key: str,
    value: Any,
    defaults: Mapping[str, Any],
    *,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> Any:
    """Match override type to default config entry."""
    if key not in defaults:
        raise ValueError(f"Unknown configuration key '{key}'.")

    default = defaults[key]

    if isinstance(default, list):
        if isinstance(value, (list, tuple)):
            return [copy.deepcopy(v) for v in value]
        return [value]

    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in boolean_true:
                return True
            if lowered in boolean_false:
                return False
            raise ValueError(f"Cannot parse boolean for '{key}': {value}")
        return bool(value)

    if isinstance(default, int) and not isinstance(default, bool):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            return int(value)

    if isinstance(default, float):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return float(value)

    if isinstance(default, str):
        if value is None:
            return None
        return str(value)

    return copy.deepcopy(value)


def update_config_value(
    config: Dict[str, Any],
    key: str,
    value: Any,
    defaults: Mapping[str, Any],
    *,
    prevalidated: bool = False,
    boolean_true: Iterable[str] = BOOLEAN_TRUE,
    boolean_false: Iterable[str] = BOOLEAN_FALSE,
) -> None:
    """Update configuration dict, coercing value types unless already validated."""
    if key not in defaults:
        raise ValueError(f"Unknown configuration key '{key}'.")
    if not prevalidated:
        value = coerce_config_value(
            key,
            value,
            defaults,
            boolean_true=boolean_true,
            boolean_false=boolean_false,
        )
    if isinstance(value, (list, dict)):
        config[key] = copy.deepcopy(value)
    else:
        config[key] = value


def parse_assignment(entry: str) -> tuple[str, str]:
    if '=' not in entry:
        raise ValueError(f"Assignments must be in key=value format, got '{entry}'.")
    key, raw = entry.split('=', 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid assignment '{entry}'.")
    return key, raw.strip()


def select_device_with_min_memory(min_memory_gb: float = 2.0, use_multiple_gpus: bool = True) -> str:
    """
    Select device using GPUs with sufficient free memory.
    
    If use_multiple_gpus is False, selects the best single GPU.
    If use_multiple_gpus is True and multiple GPUs available, selects all of them
    and returns "cuda:0" (which maps to the GPU with most free memory).
    Sets CUDA_VISIBLE_DEVICES to restrict to selected GPU(s), ordered by free memory.
    
    Args:
        min_memory_gb: Minimum free memory in GB required
        use_multiple_gpus: If True, use all available GPUs instead of just the best one
    
    Returns:
        Device string ("cpu", "cuda:X", or "cuda:0" for multi-GPU with best GPU as primary)
    """
    available_gpus = get_available_gpus(min_free_gb=min_memory_gb)
    
    if len(available_gpus) == 0:
        print(f"No GPU with at least {min_memory_gb} GB free memory found.")
        print("Falling back to CPU")
        return "cpu"
    elif len(available_gpus) == 1 or not use_multiple_gpus:
        # Select single GPU (best if multiple available)
        if len(available_gpus) == 1:
            gpu_id = available_gpus[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            print(f"Selected GPU {gpu_id}")
            return f"cuda:{gpu_id}"
        else:
            # Select the GPU with most free memory
            max_free_memory = 0
            best_gpu = None
            
            for gpu_id in available_gpus:
                free_memory, _ = torch.cuda.mem_get_info(gpu_id)
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu = gpu_id
            
            if best_gpu is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
                print(f"Selected best GPU {best_gpu} from {available_gpus} with {max_free_memory / 1e9:.2f} GB free")
                return f"cuda:{best_gpu}"
            else:
                print("Unexpected error selecting best GPU")
                return "cpu"
    else:
        # Use multiple GPUs - sort by free memory (highest first)
        gpu_memory_pairs = []
        for gpu_id in available_gpus:
            free_memory, _ = torch.cuda.mem_get_info(gpu_id)
            gpu_memory_pairs.append((gpu_id, free_memory))
        
        # Sort by free memory descending
        gpu_memory_pairs.sort(key=lambda x: x[1], reverse=True)
        sorted_gpus = [gpu_id for gpu_id, _ in gpu_memory_pairs]
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, sorted_gpus))
        best_gpu = sorted_gpus[0]
        best_memory = gpu_memory_pairs[0][1]
        print(f"Selected multiple GPUs: {sorted_gpus}")
        print(f"Primary GPU (cuda:0) is GPU {best_gpu} with {best_memory / 1e9:.2f} GB free memory")
        return f"cuda:0"  # PyTorch will see them as cuda:0, cuda:1, etc. with cuda:0 having most memory


__all__ = [
    'BOOLEAN_TRUE',
    'BOOLEAN_FALSE',
    'load_experiment_configs',
    'parse_scalar',
    'coerce_config_value',
    'update_config_value',
    'parse_assignment',
    'select_device_with_min_memory',
]
