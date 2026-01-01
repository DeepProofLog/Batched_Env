"""
Minimal Unification Engine for Evaluation.

Trace-friendly implementation that compiles cleanly with torch.compile(fullgraph=True).
No nested compilation - designed to be part of a single compiled step+reset function.
"""
import torch
from torch import Tensor
from typing import Tuple, Optional


class EngineMinimal:
    """
    Minimal engine for evaluation. 
    
    get_derived() is NOT compiled separately - it's designed to be traced
    as part of a larger compiled function (step+reset).
    """
    
    def __init__(
        self,
        facts_idx: Tensor,           # [F, 3]
        rules_idx: Tensor,           # [R, body_size, 3]
        rules_heads_idx: Tensor,     # [R, 3]
        rule_lens: Tensor,           # [R]
        padding_idx: int,
        constant_no: int,
        true_pred_idx: Optional[int],
        false_pred_idx: Optional[int],
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
        
        # Store tensors
        self.facts = facts_idx.to(device)
        self.rules = rules_idx.to(device) if rules_idx is not None else None
        self.rule_heads = rules_heads_idx.to(device) if rules_heads_idx is not None else None
        self.rule_lens = rule_lens.to(device) if rule_lens is not None else None
        
        # Build fact index by predicate
        self._build_fact_index()
        
        # Pre-allocate terminal atoms
        self.true_atom = torch.tensor([true_pred_idx, padding_idx, padding_idx], dtype=torch.long, device=device) if true_pred_idx else None
        self.false_atom = torch.tensor([false_pred_idx, padding_idx, padding_idx], dtype=torch.long, device=device) if false_pred_idx else None
    
    def _build_fact_index(self):
        """Build [P, max_facts, 3] index for O(1) fact lookup."""
        facts = self.facts
        pad = self.pad
        device = self.device
        
        if facts.numel() == 0:
            self.fact_blocks = torch.full((1, 1, 3), pad, dtype=torch.long, device=device)
            self.max_facts_per_pred = 1
            return
        
        preds = facts[:, 0]
        num_preds = int(preds.max().item()) + 2
        counts = torch.bincount(preds.long(), minlength=num_preds)
        
        max_k = min(int(counts.max().item()), 2048)
        max_k = max(max_k, 1)
        self.max_facts_per_pred = max_k
        
        self.fact_blocks = torch.full((num_preds, max_k, 3), pad, dtype=torch.long, device=device)
        
        # Sort and fill
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
        Compute derived states for current states.
        
        Trace-friendly: pure tensor operations, no Python control flow.
        
        Returns:
            derived: [B, K, A, 3] - derived states
            counts: [B] - number of valid derived states
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
        
        # Terminal check
        is_empty = (query_preds == pad)
        is_true = (query_preds == self.true_pred_idx) if self.true_pred_idx else torch.zeros_like(is_empty)
        is_false = (query_preds == self.false_pred_idx) if self.false_pred_idx else torch.zeros_like(is_empty)
        is_terminal = is_empty | is_true | is_false
        
        # Allocate output
        derived = torch.full((B, K, A, 3), pad, dtype=torch.long, device=device)
        counts = torch.zeros(B, dtype=torch.long, device=device)
        
        # Handle terminals
        if self.true_atom is not None:
            derived[:, 0, 0, :] = torch.where(is_true.unsqueeze(-1), self.true_atom.expand(B, -1), derived[:, 0, 0, :])
            counts = torch.where(is_true, torch.ones_like(counts), counts)
        
        if self.false_atom is not None:
            derived[:, 0, 0, :] = torch.where((is_false | is_empty).unsqueeze(-1), self.false_atom.expand(B, -1), derived[:, 0, 0, :])
            counts = torch.where(is_false | is_empty, torch.ones_like(counts), counts)
        
        # ===== Fact Unification =====
        # O(1) lookup via fact_blocks
        safe_preds = query_preds.clamp(0, self.fact_blocks.shape[0] - 1)
        candidate_facts = self.fact_blocks[safe_preds.long()]  # [B, max_k, 3]
        max_k = candidate_facts.shape[1]
        
        # Check predicate match
        pred_match = (candidate_facts[:, :, 0] == queries[:, 0:1])
        not_padding = (candidate_facts[:, :, 0] != pad)
        
        # Unify query with facts: check if args can unify
        # For simplicity, check exact match or variable binding
        q_args = queries[:, 1:3].unsqueeze(1)  # [B, 1, 2]
        f_args = candidate_facts[:, :, 1:3]  # [B, max_k, 2]
        
        q_is_var = q_args >= self.constant_no
        f_is_var = f_args >= self.constant_no
        args_match = (q_args == f_args) | q_is_var | f_is_var
        unify_ok = args_match.all(dim=-1) & pred_match & not_padding
        
        # Active check
        active = ~is_terminal
        fact_success = unify_ok & active.unsqueeze(1)
        
        # Build derived states from successful fact unifications
        # For each success, derived state = remaining atoms
        # (simplified: just copy remaining, actual impl would apply substitutions)
        fact_count = fact_success.sum(dim=1).clamp(max=K)
        
        # Simple: put remaining in first slots where facts succeeded
        # For correctness, need proper substitution - this is simplified
        arange_K = torch.arange(K, device=device)
        valid_mask = arange_K.unsqueeze(0) < fact_count.unsqueeze(1)
        
        # Copy remaining to derived where valid
        remaining_padded = torch.full((B, A, 3), pad, dtype=torch.long, device=device)
        remaining_padded[:, :G, :] = remaining
        
        derived = torch.where(
            valid_mask.unsqueeze(-1).unsqueeze(-1) & active.view(B, 1, 1, 1),
            remaining_padded.unsqueeze(1).expand(-1, K, -1, -1),
            derived
        )
        
        # Update counts
        counts = torch.where(active, fact_count, counts)
        
        # Ensure at least 1 for active non-terminal with no derivations (mark as false)
        no_derivations = (counts == 0) & active
        counts = torch.where(no_derivations, torch.ones_like(counts), counts)
        if self.false_atom is not None:
            derived[:, 0, 0, :] = torch.where(
                no_derivations.unsqueeze(-1),
                self.false_atom.expand(B, -1),
                derived[:, 0, 0, :]
            )
        
        return derived, counts
    
    @classmethod
    def from_vectorized_engine(cls, ve, max_derived: int, padding_atoms: int):
        """Create from existing UnificationEngineVectorized."""
        return cls(
            facts_idx=ve.facts_idx,
            rules_idx=ve.rules_idx if hasattr(ve, 'rules_idx') else None,
            rules_heads_idx=ve.rules_heads_idx if hasattr(ve, 'rules_heads_idx') else None,
            rule_lens=ve.rule_lens if hasattr(ve, 'rule_lens') else None,
            padding_idx=ve.padding_idx,
            constant_no=ve.constant_no,
            true_pred_idx=ve.true_pred_idx,
            false_pred_idx=ve.false_pred_idx,
            max_derived=max_derived,
            padding_atoms=padding_atoms,
            device=ve.device,
        )
