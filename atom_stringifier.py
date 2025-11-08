from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch

LongTensor = torch.LongTensor

@dataclass(frozen=True)
class AtomStringifier:
    """
    Lightweight, immutable stringifier for debug-only use.
    Keeps small string tables; no big tensors or indices are held.
    """
    n_constants: int
    idx2constant: Sequence[str]
    idx2predicate: Sequence[str]
    idx2template_var: Sequence[str]

    def atom_to_str(self, atom_idx: LongTensor) -> str:
        p, a, b = [int(x) for x in atom_idx.tolist()]
        ps = self.idx2predicate[p] if 0 <= p < len(self.idx2predicate) else str(p)

        def term_str(t: int) -> str:
            if 1 <= t <= self.n_constants:
                return self.idx2constant[t]
            if self.n_constants < t <= self.n_constants + len(self.idx2template_var) - 1:
                tv = t - self.n_constants
                return self.idx2template_var[tv] if 0 <= tv < len(self.idx2template_var) else f"v{t}"
            return f"_{t}"

        return f"{ps}({term_str(a)},{term_str(b)})"

    def state_to_str(self, state_idx: LongTensor) -> str:
        if state_idx.numel() == 0:
            return "<empty>"
        parts = [self.atom_to_str(row) for row in state_idx]
        return ", ".join(parts)
    
    @classmethod
    def from_index_manager(cls, index_manager):
        """Create an AtomStringifier from an IndexManager."""
        return cls(
            n_constants=index_manager.constant_no,
            idx2constant=index_manager.idx2constant,
            idx2predicate=index_manager.idx2predicate,
            idx2template_var=index_manager.idx2template_var,
        )
