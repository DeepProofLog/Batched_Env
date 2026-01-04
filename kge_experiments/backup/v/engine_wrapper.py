"""
Engine Wrapper for Evaluation.

Provides a simplified interface around UnificationEngineVectorized for
use in compiled evaluation loops. Handles all the static shape processing
and validation internally.

Usage:
    engine = EngineWrapper.from_vectorized_engine(vec_engine, B, S, A)
    derived, counts = engine.get_derived(current_states)
"""
import torch
from torch import Tensor
from typing import Tuple, Optional

from unification import UnificationEngineVectorized


class EngineWrapper:
    """
    Simplified wrapper around UnificationEngineVectorized.
    
    Key features:
    - Simple get_derived(current) -> (derived, counts) interface
    - Handles static shape conversion internally
    - Pre-allocates all buffers for stable addresses
    - Compatible with torch.compile and CUDA graphs
    """
    
    def __init__(
        self,
        vec_engine: UnificationEngineVectorized,
        batch_size: int,
        padding_states: int,  # S = max derived states 
        padding_atoms: int,   # A = max atoms per state
        device: torch.device,
    ):
        self.engine = vec_engine
        self.B = batch_size
        self.S = padding_states
        self.A = padding_atoms
        self.device = device
        
        # Engine attributes
        self.pad = vec_engine.padding_idx
        self.true_pred_idx = vec_engine.true_pred_idx
        self.false_pred_idx = vec_engine.false_pred_idx
        self.constant_no = vec_engine.constant_no
        
        # Pre-allocate variable indices buffer - MUST be constant_no + 1
        runtime_var_start = vec_engine.constant_no + 1
        self._var_buf = torch.full((batch_size,), runtime_var_start, dtype=torch.long, device=device)
        
        # Pre-allocate false state template
        self._false_state = torch.full((padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        if self.false_pred_idx is not None:
            self._false_state[0, 0, 0] = self.false_pred_idx
        
        # Pre-allocate buffers for output
        self._derived_buf = torch.full((batch_size, padding_states, padding_atoms, 3), self.pad, dtype=torch.long, device=device)
        self._counts_buf = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Index tensors
        self._arange_S = torch.arange(padding_states, device=device)
        self._ones_B = torch.ones(batch_size, dtype=torch.long, device=device)
    
    def get_derived(
        self,
        current_states: Tensor,  # [B, A, 3]
        excluded_queries: Optional[Tensor] = None,  # [B, 1, 3] or None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute derived states from current states.
        
        This wraps the original engine's get_derived_states_compiled and
        handles static shape conversion.
        
        Args:
            current_states: [B, A, 3] Current states (queries at position 0)
            excluded_queries: Optional [B, 1, 3] queries to exclude (cycle prevention)
            
        Returns:
            derived: [B, S, A, 3] Derived states padded to static shape
            counts: [B] Number of valid derived states per batch element
        """
        B, S, A, pad = self.B, self.S, self.A, self.pad
        
        # Call original engine
        derived_raw, counts_raw, _ = self.engine.get_derived_states_compiled(
            current_states, self._var_buf, excluded_queries
        )
        
        # Convert to static shape [B, S, A, 3]
        derived = torch.full((B, S, A, 3), pad, dtype=torch.long, device=self.device)
        K = min(derived_raw.shape[1], S)
        M = min(derived_raw.shape[2], A)
        derived[:, :K, :M, :] = derived_raw[:, :K, :M, :]
        
        # Validate counts - ensure valid atoms and within S
        within = self._arange_S.unsqueeze(0) < counts_raw.unsqueeze(1)  # [B, S]
        valid_atom = derived[:, :, :, 0] != pad  # [B, S, A]
        atom_counts = valid_atom.sum(dim=2)  # [B, S]
        base_valid = within & (atom_counts <= A) & (atom_counts > 0)  # [B, S]
        
        # Replace invalid with false_state
        derived = torch.where(
            base_valid.unsqueeze(-1).unsqueeze(-1),
            derived,
            self._false_state.unsqueeze(0)
        )
        
        # Final counts
        counts = base_valid.sum(dim=1)  # [B]
        counts = torch.where(counts == 0, self._ones_B, counts)
        
        return derived, counts
    
    @classmethod
    def from_vectorized_engine(
        cls,
        vec_engine: UnificationEngineVectorized,
        batch_size: int,
        padding_states: int,
        padding_atoms: int,
    ):
        """Create wrapper from existing UnificationEngineVectorized."""
        return cls(
            vec_engine=vec_engine,
            batch_size=batch_size,
            padding_states=padding_states,
            padding_atoms=padding_atoms,
            device=vec_engine.device,
        )
    
    def compile(self, mode: str = 'reduce-overhead'):
        """Compile get_derived for faster execution."""
        self._compiled_get_derived = torch.compile(
            self.get_derived, 
            mode=mode, 
            fullgraph=True, 
            dynamic=False
        )
        print(f"[EngineWrapper] Compiled (mode={mode})")
    
    def get_derived_compiled(
        self,
        current_states: Tensor,
        excluded_queries: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Use compiled version if available."""
        if hasattr(self, '_compiled_get_derived'):
            return self._compiled_get_derived(current_states, excluded_queries)
        return self.get_derived(current_states, excluded_queries)
