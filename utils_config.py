import ast
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


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


__all__ = [
    'BOOLEAN_TRUE',
    'BOOLEAN_FALSE',
    'load_experiment_configs',
    'parse_scalar',
    'coerce_config_value',
    'update_config_value',
    'parse_assignment',
]
