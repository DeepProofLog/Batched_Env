"""Unified KGE inference wrapper for PyTorch and PyKEEN backends.

This module provides a single entrypoint to load a KGE checkpoint from either
`kge_pytorch/models` or `kge_pykeen/models` and exposes a shared `predict_batch`
API for downstream fusion with RL evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple
import warnings


_BACKEND_ALIASES = {
    "torch": "pytorch",
    "pytorch": "pytorch",
    "pykeen": "pykeen",
}
_BACKEND_PREFIX = {
    "pytorch": "torch",
    "pykeen": "pykeen",
}


def normalize_backend(backend: Optional[str]) -> str:
    """Normalize backend name and validate supported options."""
    if backend is None:
        return "pytorch"
    normalized = backend.strip().lower()
    if normalized in _BACKEND_ALIASES:
        return _BACKEND_ALIASES[normalized]
    raise ValueError(
        f"Unsupported KGE backend '{backend}'. Use 'pytorch' or 'pykeen'."
    )


def default_checkpoint_dir(backend: str) -> str:
    """Return default checkpoint directory for a backend."""
    base_dir = Path(__file__).resolve().parent
    folder = "kge_pytorch" if backend == "pytorch" else "kge_pykeen"
    return str(base_dir / folder / "models")


def find_latest_run(checkpoint_dir: str, prefix: Optional[str] = None) -> Optional[str]:
    """Find the most recent run directory under the checkpoint root."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if prefix:
        candidates = [p for p in candidates if p.name.startswith(prefix)]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def _get_backend_class(backend: str) -> Tuple[type, str]:
    """Load the appropriate KGE backend class based on the specified backend."""
    backend = normalize_backend(backend)
    if backend == "pytorch":
        from kge_pytorch.kge_inference_torch import KGEInference as BackendKGEInference
        return BackendKGEInference, "pytorch"
    if backend == "pykeen":
        from kge_pykeen.kge_inference_pykeen import KGEInference as BackendKGEInference
        return BackendKGEInference, "pykeen"
    raise ValueError(f"Unsupported KGE backend '{backend}'.")


class KGEInference:
    """Wrapper class that delegates to the appropriate backend implementation."""

    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        checkpoint_dir: str,
        run_signature: str,
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        backend: str = "pytorch",
        **kwargs: Any,
    ) -> None:
        BackendClass, self.backend = _get_backend_class(backend)
        self._backend_engine = BackendClass(
            dataset_name=dataset_name,
            base_path=base_path,
            checkpoint_dir=checkpoint_dir,
            run_signature=run_signature,
            seed=seed,
            scores_file_path=scores_file_path,
            **kwargs,
        )
        print(f"KGE Engine initialized with backend: {self.backend}")

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the backend engine."""
        return getattr(self._backend_engine, name)

    def __repr__(self) -> str:
        return f"KGEInference(backend={self.backend}, engine={self._backend_engine})"


def build_kge_inference(
    config: Any,
    *,
    index_manager: Optional[Any] = None,
) -> Optional[KGEInference]:
    """Build a KGE inference engine from a TrainConfig-like object."""
    if not bool(getattr(config, "kge_inference", False)):
        return None

    backend = normalize_backend(getattr(config, "kge_engine", None))
    checkpoint_dir = getattr(config, "kge_checkpoint_dir", None) or default_checkpoint_dir(backend)
    run_signature = getattr(config, "kge_run_signature", None)
    if not run_signature:
        prefix = f"{_BACKEND_PREFIX[backend]}_{getattr(config, 'dataset', '')}_"
        run_signature = find_latest_run(checkpoint_dir, prefix=prefix)
        if run_signature is None:
            raise ValueError(
                "kge_run_signature is required when no checkpoints are found. "
                f"Expected checkpoints under {checkpoint_dir}"
            )
        warnings.warn(
            f"kge_run_signature not provided; using latest run '{run_signature}'.",
            RuntimeWarning,
            stacklevel=2,
        )

    engine = KGEInference(
        dataset_name=getattr(config, "dataset", ""),
        base_path=getattr(config, "data_path", ""),
        checkpoint_dir=checkpoint_dir,
        run_signature=run_signature,
        seed=int(getattr(config, "seed", 0)),
        scores_file_path=getattr(config, "kge_scores_file", None),
        backend=backend,
        device=str(getattr(config, "device", "cpu")),
    )

    if index_manager is not None:
        setattr(engine, "index_manager", index_manager)
    return engine


def current_backend(engine: Optional[KGEInference] = None) -> str:
    """Return the name of the active backend implementation."""
    if engine is not None:
        return engine.backend
    return "pytorch"


__all__ = [
    "KGEInference",
    "build_kge_inference",
    "current_backend",
    "default_checkpoint_dir",
    "find_latest_run",
    "normalize_backend",
]
