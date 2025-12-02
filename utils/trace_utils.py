import json
from pathlib import Path
from typing import Any, Dict, Optional


def _to_python_scalar(x: Any):
    """Best-effort conversion of tensors/arrays to plain Python scalars or lists."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.item()
            return x.tolist()
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return x
    return x


class TraceRecorder:
    """Simple JSONL trace recorder for rollout/eval parity debugging."""

    def __init__(self, trace_dir: str, prefix: str):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.records = []

    def log_step(self, **kwargs):
        rec = {k: _to_python_scalar(v) for k, v in kwargs.items()}
        self.records.append(rec)

    def log_eval(self, split: str, metrics: Dict[str, Any]):
        self.records.append({"phase": "eval", "split": split, "metrics": metrics})

    def flush(self, filename: Optional[str] = None):
        if filename is None:
            filename = f"{self.prefix}_trace.jsonl"
        path = self.trace_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec))
                f.write("\n")
        return path