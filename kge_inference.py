"""Compatibility wrapper for the KGE inference engine.

The training and evaluation code expects to import ``KGEInference`` from this
module.  The actual implementation can be from ``kge_tf``, ``kge_pytorch``, or 
``kge_pykeen`` depending on the backend parameter passed during initialization.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional, Any


def _get_backend_class(backend: str):
    """Load the appropriate KGE backend class based on the specified backend."""
    backend = backend.strip().lower()
    
    if backend in {"", "tf", "tensorflow"}:
        from kge_tf.kge_inference_tf import KGEInference as BackendKGEInference
        return BackendKGEInference, "tensorflow"
    elif backend in {"torch", "pytorch"}:
        try:
            from kge_pytorch.kge_inference_torch import KGEInference as BackendKGEInference
            return BackendKGEInference, "pytorch"
        except ImportError as e:
            warnings.warn(
                f"PyTorch KGE backend could not be imported: {e}. Falling back to TensorFlow.",
                RuntimeWarning,
                stacklevel=2,
            )
            from kge_tf.kge_inference_tf import KGEInference as BackendKGEInference
            return BackendKGEInference, "tensorflow"
    elif backend in {"pykeen"}:
        try:
            from kge_pykeen.kge_inference_pykeen import KGEInference as BackendKGEInference
            return BackendKGEInference, "pykeen"
        except ImportError as e:
            warnings.warn(
                f"PyKEEN KGE backend could not be imported: {e}. Falling back to TensorFlow.",
                RuntimeWarning,
                stacklevel=2,
            )
            from kge_tf.kge_inference_tf import KGEInference as BackendKGEInference
            return BackendKGEInference, "tensorflow"
    else:
        warnings.warn(
            f"Unknown KGE backend '{backend}'. Falling back to TensorFlow.",
            RuntimeWarning,
            stacklevel=2,
        )
        from kge_tf.kge_inference_tf import KGEInference as BackendKGEInference
        return BackendKGEInference, "tensorflow"


class KGEInference:
    """Wrapper class that delegates to the appropriate KGE backend implementation."""
    
    def __init__(
        self,
        dataset_name: str,
        base_path: str,
        checkpoint_dir: str,
        run_signature: str,
        seed: int = 0,
        scores_file_path: Optional[str] = None,
        backend: str = "tf",
        **kwargs
    ):
        """
        Initialize KGE inference engine with the specified backend.
        
        Args:
            dataset_name: Name of the dataset
            base_path: Base path to the data directory
            checkpoint_dir: Directory containing KGE model checkpoints
            run_signature: Signature identifying the specific KGE model run
            seed: Random seed
            scores_file_path: Optional path to pre-computed scores file
            backend: KGE backend to use ('tf', 'pytorch', or 'pykeen')
            **kwargs: Additional backend-specific arguments
        """
        BackendClass, self.backend = _get_backend_class(backend)
        
        # Initialize the backend-specific implementation
        self._backend_engine = BackendClass(
            dataset_name=dataset_name,
            base_path=base_path,
            checkpoint_dir=checkpoint_dir,
            run_signature=run_signature,
            seed=seed,
            scores_file_path=scores_file_path,
            **kwargs
        )
        
        print(f"KGE Engine initialized with backend: {self.backend}")
    
    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the backend engine."""
        return getattr(self._backend_engine, name)
    
    def __repr__(self) -> str:
        return f"KGEInference(backend={self.backend}, engine={self._backend_engine})"


def current_backend(engine: Optional[KGEInference] = None) -> str:
    """Return the name of the active backend implementation."""
    if engine is not None:
        return engine.backend
    # Default backend when no engine is provided
    return "tensorflow"


__all__ = ["KGEInference", "current_backend"]
