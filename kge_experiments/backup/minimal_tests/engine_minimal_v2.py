"""
Minimal Unification Engine V2 for Evaluation.

NO Python control flow - pure tensor operations for CUDA graph capture.
"""
import torch
from torch import Tensor
from typing import Tuple


class EngineMinimalV2:
    """
    Engine with zero Python control flow - fully compatible with CUDA graph capture.
    """
    
    def __init__(
        self,
        facts_idx: Tensor,           # [F, 3]
        padding_idx: int,
        constant_no: int,
        true_pred_idx: int,          # Must be provided (use -1 if none)
        false_pred_idx: int,         # Must be provided (use -1 if none)
        max_derived: int,            # K_max
        padding_atoms: int,          # A
        device: torch.device,
    ):
        self.device = device
        self.pad = padding_idx
        self.constant_no = constant_no
        self.true_pred_idx = true_pred_idx
        self.false_pred_idx = false_pred_idx
        self.K = max_derived
        self.A = padding_atoms
        
        self.facts = facts_idx.to(device)
        self._build_fact_index()
        
        # Pre-allocate terminal atoms (always exists, use pad if not used)
        self.true_atom = torch.tensor([true_pred_idx, padding_idx, padding_idx], dtype=torch.long, device=device)
        self.false_atom = torch.tensor([false_pred_idx, padding_idx, padding_idx], dtype=torch.long, device=device)
    
    def _build_fact_index(self):
        """Build [P, max_facts, 3] index for O(1) fact lookup."""
        facts = self.facts
        pad = self.pad
        device = self.device
        
        preds = facts[:, 0]
        num_preds = max(int(preds.max().item()) + 2, 1) if facts.numel() > 0 else 1
        counts = torch.bincount(preds.long(), minlength=num_preds) if facts.numel() > 0 else torch.zeros(1, dtype=torch.long, device=device)
        
        max_k = min(max(int(counts.max().item()), 1), 2048) if counts.numel() > 0 else 1
        self.max_facts_per_pred = max_k
        
        self.fact_blocks = torch.full((num_preds, max_k, 3), pad, dtype=torch.long, device=device)
        
        if facts.numel() > 0:
            order = torch.argsort(preds, stable=True)
            facts_sorted = facts[order]
            preds_sorted = preds[order]
            
            group_starts = torch.zeros(num_preds + 1, dtype=torch.long, device=device)
            group_starts[1:] = counts.cumsum(0)
            
            global_idx = torch.arange(facts.shape[0], device=device)
            local_idx = global_idx - group_starts[preds_sorted.long()]
            
            valid = local_idx < max_k
            local_idx_clamped = local_idx.clamp(max=max_k - 1)
            
            self.fact_blocks[preds_sorted[valid].long(), local_idx_clamped[valid]] = facts_sorted[valid]
    
    def get_derived(
        self,
        current: Tensor,  # [B, A, 3]
    ) -> Tuple[Tensor, Tensor]:
        """
        NO Python control flow - pure tensor ops.
        """
        B, A, _ = current.shape
        K = self.K
        pad = self.pad
        device = self.device
        
        # Extract query (first atom)
        queries = current[:, 0, :]  # [B, 3]
        remaining = current[:, 1:, :]  # [B, A-1, 3]
        G = remaining.shape[1]
        
        query_preds = queries[:, 0]
        
        # Terminal check - pure tensor ops, no Python if
        is_empty = (query_preds == pad)
        is_true = (query_preds == self.true_pred_idx)
        is_false = (query_preds == self.false_pred_idx)
        is_terminal = is_empty | is_true | is_false
        
        # Allocate output
        derived = torch.full((B, K, A, 3), pad, dtype=torch.long, device=device)
        counts = torch.zeros(B, dtype=torch.long, device=device)
        
        # Handle terminals - torch.where, no Python if
        derived[:, 0, 0, :] = torch.where(
            is_true.unsqueeze(-1), 
            self.true_atom.unsqueeze(0).expand(B, -1), 
            derived[:, 0, 0, :]
        )
        counts = torch.where(is_true, torch.ones_like(counts), counts)
        
        derived[:, 0, 0, :] = torch.where(
            (is_false | is_empty).unsqueeze(-1), 
            self.false_atom.unsqueeze(0).expand(B, -1), 
            derived[:, 0, 0, :]
        )
        counts = torch.where(is_false | is_empty, torch.ones_like(counts), counts)
        
        # Fact Unification - O(1) lookup
        safe_preds = query_preds.clamp(0, self.fact_blocks.shape[0] - 1)
        candidate_facts = self.fact_blocks[safe_preds.long()]  # [B, max_k, 3]
        max_k = candidate_facts.shape[1]
        
        pred_match = (candidate_facts[:, :, 0] == queries[:, 0:1])
        not_padding = (candidate_facts[:, :, 0] != pad)
        
        q_args = queries[:, 1:3].unsqueeze(1)  # [B, 1, 2]
        f_args = candidate_facts[:, :, 1:3]  # [B, max_k, 2]
        
        q_is_var = q_args >= self.constant_no
        f_is_var = f_args >= self.constant_no
        args_match = (q_args == f_args) | q_is_var | f_is_var
        unify_ok = args_match.all(dim=-1) & pred_match & not_padding
        
        active = ~is_terminal
        fact_success = unify_ok & active.unsqueeze(1)
        
        fact_count = fact_success.sum(dim=1).clamp(max=K)
        
        arange_K = torch.arange(K, device=device)
        valid_mask = arange_K.unsqueeze(0) < fact_count.unsqueeze(1)
        
        remaining_padded = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        remaining_padded[:, :G, :] = remaining
        
        derived = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1) & active.view(B, 1, 1, 1),
            remaining_padded.unsqueeze(1).expand(-1, K, -1, -1),
            derived
        )
        
        counts = torch.where(active, fact_count, counts)
        
        # Handle no derivations - torch.where, no Python if
        no_derivations = (counts == 0) & active
        counts = torch.where(no_derivations, torch.ones_like(counts), counts)
        derived[:, 0, 0, :] = torch.where(
            no_derivations.unsqueeze(-1),
            self.false_atom.unsqueeze(0).expand(B, -1),
            derived[:, 0, 0, :]
        )
        
        return derived, counts
    
    @classmethod
    def from_vectorized_engine(cls, ve, max_derived: int, padding_atoms: int):
        """Create from existing UnificationEngineVectorized."""
        return cls(
            facts_idx=ve.facts_idx,
            padding_idx=ve.padding_idx,
            constant_no=ve.constant_no,
            true_pred_idx=ve.true_pred_idx if ve.true_pred_idx is not None else ve.padding_idx,
            false_pred_idx=ve.false_pred_idx if ve.false_pred_idx is not None else ve.padding_idx,
            max_derived=max_derived,
            padding_atoms=padding_atoms,
            device=ve.device,
        )
