"""
Compilable variable enumeration for DP prover.

This module provides functions to enumerate all possible ground instantiations
of variable queries. Used when proving queries like `locatedInCR(X, europe)`.

Key Insight:
    For `locatedInCR(X, europe)`:
    1. Get all facts with predicate `locatedInCR`
    2. Filter by fixed argument (arg1 == europe_idx)
    3. Extract variable positions as candidate bindings
    4. Batch-prove each binding

Tensor Shapes:
    - query: [3] with variables (idx > constant_no)
    - candidates: [K, 3] ground queries to prove
    - bindings: [K, num_vars] variable assignments
"""

from __future__ import annotations
from typing import Tuple, Optional, List
import torch
from torch import Tensor


@torch.no_grad()
def identify_variables(
    query: Tensor,      # [3] or [B, 3]
    constant_no: int,
    padding_idx: int = 0,
) -> Tuple[Tensor, Tensor]:
    """
    Identify which positions in query contain variables.

    Args:
        query: [3] or [B, 3] query atoms
        constant_no: Maximum constant index

    Returns:
        is_var: [3] or [B, 3] boolean mask (excluding predicate position)
        var_positions: [num_vars] positions (1 or 2) containing variables
    """
    if query.dim() == 1:
        query = query.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    # Variables have index > constant_no and != padding
    is_var = (query > constant_no) & (query != padding_idx)  # [B, 3]

    # Predicate (position 0) cannot be a variable
    is_var[:, 0] = False

    if squeeze:
        is_var = is_var.squeeze(0)

    # Get variable positions (arg positions 1, 2)
    args_is_var = is_var[..., 1:]  # [..., 2]
    var_positions = []
    if args_is_var[..., 0].any():
        var_positions.append(1)
    if args_is_var[..., 1].any():
        var_positions.append(2)

    return is_var, torch.tensor(var_positions, dtype=torch.long, device=query.device)


@torch.no_grad()
def enumerate_bindings_from_facts(
    query: Tensor,              # [3] with variables
    facts: Tensor,              # [F, 3] all facts
    constant_no: int,
    padding_idx: int = 0,
    max_bindings: int = 1000,
) -> Tuple[Tensor, Tensor, int]:
    """
    Find all ground instantiations of a variable query from facts.

    For `locatedInCR(X, europe)`:
    1. Filter facts by predicate = locatedInCR
    2. Filter by ground args (arg1 == europe)
    3. Extract values for variable positions as candidates

    Args:
        query: [3] query with variables (idx > constant_no)
        facts: [F, 3] all known facts
        constant_no: Max constant index
        padding_idx: Padding value
        max_bindings: Max candidates to return

    Returns:
        candidates: [K, 3] ground query instantiations (K <= max_bindings)
        bindings: [K, 2] variable bindings [(var_pos, value), ...]
        count: number of valid candidates
    """
    device = query.device
    F = facts.shape[0]

    if F == 0:
        return (
            torch.zeros(0, 3, dtype=torch.long, device=device),
            torch.zeros(0, 2, dtype=torch.long, device=device),
            0,
        )

    pred = query[0]
    arg0 = query[1]
    arg1 = query[2]

    is_var0 = arg0 > constant_no
    is_var1 = arg1 > constant_no

    # Filter facts by predicate
    pred_match = facts[:, 0] == pred  # [F]

    # Filter by ground arguments
    if not is_var0:
        pred_match = pred_match & (facts[:, 1] == arg0)
    if not is_var1:
        pred_match = pred_match & (facts[:, 2] == arg1)

    # Get matching facts
    matching_facts = facts[pred_match]  # [K, 3]
    K = matching_facts.shape[0]

    if K == 0:
        return (
            torch.zeros(0, 3, dtype=torch.long, device=device),
            torch.zeros(0, 2, dtype=torch.long, device=device),
            0,
        )

    # Limit to max_bindings
    K = min(K, max_bindings)
    matching_facts = matching_facts[:K]

    # Build ground candidates
    candidates = matching_facts.clone()  # [K, 3]

    # Build bindings tensor
    # Each binding is (var_position, constant_value)
    bindings_list = []
    if is_var0:
        bindings_list.append(torch.stack([
            torch.ones(K, dtype=torch.long, device=device),  # position 1
            matching_facts[:, 1]  # values
        ], dim=1))
    if is_var1:
        bindings_list.append(torch.stack([
            torch.full((K,), 2, dtype=torch.long, device=device),  # position 2
            matching_facts[:, 2]  # values
        ], dim=1))

    if bindings_list:
        bindings = torch.cat(bindings_list, dim=0)
    else:
        bindings = torch.zeros(0, 2, dtype=torch.long, device=device)

    return candidates, bindings, K


@torch.no_grad()
def enumerate_bindings_batch(
    queries: Tensor,            # [B, 3] queries with variables
    facts: Tensor,              # [F, 3] all facts
    constant_no: int,
    padding_idx: int = 0,
    max_bindings_per_query: int = 100,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Batch enumeration of variable query bindings.

    Args:
        queries: [B, 3] queries (may have variables)
        facts: [F, 3] all facts
        constant_no: Max constant index
        padding_idx: Padding value
        max_bindings_per_query: Max candidates per query

    Returns:
        candidates: [B, max_bindings, 3] ground instantiations (padded)
        valid_mask: [B, max_bindings] which candidates are valid
        counts: [B] number of valid candidates per query
    """
    B = queries.shape[0]
    device = queries.device
    pad = padding_idx
    max_K = max_bindings_per_query

    candidates = torch.full((B, max_K, 3), pad, dtype=torch.long, device=device)
    valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)
    counts = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(B):
        query = queries[i]
        cands, _, count = enumerate_bindings_from_facts(
            query, facts, constant_no, padding_idx, max_K
        )

        if count > 0:
            candidates[i, :count] = cands
            valid_mask[i, :count] = True
            counts[i] = count

    return candidates, valid_mask, counts


@torch.no_grad()
def enumerate_entities_for_position(
    pred_idx: int,
    position: int,              # 1 or 2 (arg0 or arg1)
    fixed_value: Optional[int], # Value for the other position
    facts: Tensor,              # [F, 3]
    max_entities: int = 1000,
) -> Tensor:
    """
    Get all entities that appear at a given position for a predicate.

    Useful for domain/range enumeration.

    Args:
        pred_idx: Predicate index
        position: 1 for arg0, 2 for arg1
        fixed_value: If provided, filter by the other argument
        facts: [F, 3] all facts
        max_entities: Max entities to return

    Returns:
        entities: [K] unique entity indices at the position
    """
    device = facts.device

    # Filter by predicate
    mask = facts[:, 0] == pred_idx

    # Filter by fixed argument if provided
    if fixed_value is not None:
        other_pos = 3 - position  # 1->2, 2->1
        mask = mask & (facts[:, other_pos] == fixed_value)

    matching_facts = facts[mask]

    if matching_facts.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    # Extract entities at position
    entities = matching_facts[:, position]

    # Get unique entities
    unique_entities = torch.unique(entities)

    # Limit count
    if unique_entities.shape[0] > max_entities:
        unique_entities = unique_entities[:max_entities]

    return unique_entities


class BindingEnumerator:
    """
    Efficient binding enumeration with caching.

    Caches predicate-filtered facts for faster repeated queries.
    """

    def __init__(
        self,
        facts: Tensor,
        constant_no: int,
        padding_idx: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize enumerator.

        Args:
            facts: [F, 3] all facts
            constant_no: Max constant index
            padding_idx: Padding value
            device: Torch device
        """
        self.facts = facts
        self.constant_no = constant_no
        self.padding_idx = padding_idx
        self.device = device or facts.device

        # Build predicate index
        self._build_predicate_index()

    def _build_predicate_index(self) -> None:
        """Index facts by predicate for O(1) lookup."""
        if self.facts.numel() == 0:
            self.pred_facts = {}
            return

        self.pred_facts = {}
        preds = self.facts[:, 0]
        unique_preds = torch.unique(preds)

        for pred in unique_preds:
            pred_val = pred.item()
            mask = preds == pred
            self.pred_facts[pred_val] = self.facts[mask]

    @torch.no_grad()
    def enumerate(
        self,
        query: Tensor,
        max_bindings: int = 1000,
    ) -> Tuple[Tensor, int]:
        """
        Enumerate ground instantiations for a query.

        Args:
            query: [3] query (may have variables)
            max_bindings: Max instantiations

        Returns:
            candidates: [K, 3] ground queries
            count: number of valid candidates
        """
        pred = query[0].item()
        arg0 = query[1].item()
        arg1 = query[2].item()

        is_var0 = arg0 > self.constant_no
        is_var1 = arg1 > self.constant_no

        # Get facts for this predicate
        pred_facts = self.pred_facts.get(pred)
        if pred_facts is None or pred_facts.shape[0] == 0:
            return torch.zeros(0, 3, dtype=torch.long, device=self.device), 0

        # Filter by ground arguments
        mask = torch.ones(pred_facts.shape[0], dtype=torch.bool, device=self.device)
        if not is_var0:
            mask = mask & (pred_facts[:, 1] == arg0)
        if not is_var1:
            mask = mask & (pred_facts[:, 2] == arg1)

        matching = pred_facts[mask]
        K = min(matching.shape[0], max_bindings)

        return matching[:K], K

    @torch.no_grad()
    def enumerate_batch(
        self,
        queries: Tensor,
        max_bindings_per_query: int = 100,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Batch enumeration with padding.

        Args:
            queries: [B, 3] queries
            max_bindings_per_query: Max per query

        Returns:
            candidates: [B, max_K, 3] padded
            valid_mask: [B, max_K]
            counts: [B]
        """
        B = queries.shape[0]
        max_K = max_bindings_per_query
        pad = self.padding_idx
        device = self.device

        candidates = torch.full((B, max_K, 3), pad, dtype=torch.long, device=device)
        valid_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)
        counts = torch.zeros(B, dtype=torch.long, device=device)

        for i in range(B):
            cands, count = self.enumerate(queries[i], max_K)
            if count > 0:
                candidates[i, :count] = cands
                valid_mask[i, :count] = True
                counts[i] = count

        return candidates, valid_mask, counts
