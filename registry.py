import sys
from pathlib import Path
from typing import Any, Dict

# Automatically set up paths for experiments
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "kge_experiments") not in sys.path:
    sys.path.insert(0, str(_ROOT / "kge_experiments"))

# Registry of experiment builders
_BUILDERS: Dict[str, Any] = {}


def register(name: str, builder_module):
    """Register an experiment builder module."""
    _BUILDERS[name] = builder_module


def get_builder(name: str):
    """Get builder for experiment type or raise with available options."""
    if name not in _BUILDERS:
        available = list(_BUILDERS.keys())
        raise ValueError(
            f"Unknown experiment type: '{name}'. "
            f"Available types: {available}"
        )
    return _BUILDERS[name]


def available_experiments():
    """Return list of registered experiment types."""
    return list(_BUILDERS.keys())


# =============================================================================
# Public API Functions
# =============================================================================

def load_config(experiment: str, **overrides):
    """Load default config for an experiment type."""
    builder = get_builder(experiment)
    return builder.get_default_config(**overrides)


def build_env(config):
    """Build environment from config."""
    exp_type = getattr(config, 'experiment_type', None)
    if exp_type is None:
        raise ValueError("Config must have 'experiment_type' attribute. Use load_config() first.")
    builder = get_builder(exp_type)
    return builder.create_env(config)


def build_policy(config, env=None):
    """Build policy from config. Must be called after build_env()."""
    exp_type = getattr(config, 'experiment_type', None)
    if exp_type is None:
        raise ValueError("Config must have 'experiment_type' attribute. Use load_config() first.")
    builder = get_builder(exp_type)
    return builder.create_policy(config, env)


def get_algorithm(policy, env, config):
    """
    Create algorithm instance based on experiment type.
    Also sets up callbacks for training.

    Args:
        policy: Policy network
        env: Environment
        config: Configuration object with experiment_type attribute

    Returns:
        Algorithm instance (e.g., PPO for KGE experiments)
    """
    exp_type = getattr(config, 'experiment_type', None)
    if exp_type is None:
        raise ValueError("Config must have 'experiment_type' attribute. Use load_config() first.")
    builder = get_builder(exp_type)
    return builder.create_algorithm(policy, env, config)


def run_evaluation(algorithm, config):
    """
    Run evaluation using experiment-specific logic.

    Args:
        algorithm: Trained algorithm instance
        config: Configuration object with experiment_type attribute

    Returns:
        Dictionary of evaluation results
    """
    exp_type = getattr(config, 'experiment_type', None)
    if exp_type is None:
        raise ValueError("Config must have 'experiment_type' attribute.")
    builder = get_builder(exp_type)
    return builder.run_evaluation(algorithm, config)


# =============================================================================
# Auto-register known experiments on import
# =============================================================================

def _auto_register():
    """Register all known experiment builders."""
    try:
        from kge_experiments import builder as kge_builder
        register('kge', kge_builder)
    except ImportError:
        pass

_auto_register()
