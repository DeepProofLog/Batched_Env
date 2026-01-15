# Lazy imports to avoid circular dependency issues when used via registry
# Direct imports available when running from kge_experiments directory

def __getattr__(name):
    """Lazy loading of submodules."""
    if name in ('get_default_config', 'create_env', 'create_policy', 'create_algorithm', 'run_evaluation'):
        from . import builder
        return getattr(builder, name)
    if name == 'print_results':
        from .utils import print_results
        return print_results
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
