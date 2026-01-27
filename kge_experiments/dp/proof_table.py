"""
Tensor-based memoization table for DP prover.

This module provides GPU-resident proof caching with fixed tensor shapes
for torch.compile compatibility. Uses open addressing hash table.

Tensor Shapes:
    - table: [capacity, 5] = [pred, arg0, arg1, is_proven, depth]
    - occupied: [capacity] boolean mask
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
from torch import Tensor


class ProofTable:
    """
    GPU-resident memoization table for proof results.

    Uses open addressing with linear probing for collision resolution.
    All operations are vectorized for torch.compile compatibility.
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device,
        pack_base: int = 100000,
    ):
        """
        Initialize the proof table.

        Args:
            capacity: Maximum number of entries (should be ~2x expected entries)
            device: Torch device for tensors
            pack_base: Base for hashing triples
        """
        self.capacity = capacity
        self.device = device
        self.pack_base = pack_base

        # Table entries: [capacity, 5] = [pred, arg0, arg1, is_proven, depth]
        # Using long for all fields for uniform handling
        self.table = torch.zeros(capacity, 5, dtype=torch.long, device=device)

        # Occupied mask
        self.occupied = torch.zeros(capacity, dtype=torch.bool, device=device)

        # Stats for debugging
        self.hits = 0
        self.misses = 0
        self.inserts = 0

        # Mark as static for CUDA graphs
        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.table)
            torch._dynamo.mark_static_address(self.occupied)

    def _hash(self, goals: Tensor) -> Tensor:
        """
        Hash goals to table indices.

        Args:
            goals: [B, 3] ground atoms (pred, arg0, arg1)

        Returns:
            indices: [B] hash indices into table
        """
        # Simple polynomial hash: ((pred * base) + arg0) * base + arg1
        p, a, b = goals[:, 0].long(), goals[:, 1].long(), goals[:, 2].long()
        h = ((p * self.pack_base) + a) * self.pack_base + b
        return h.abs() % self.capacity

    @torch.no_grad()
    def lookup(self, goals: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Look up proof results for goals.

        Args:
            goals: [B, 3] ground atoms to look up

        Returns:
            found: [B] boolean - whether entry was found
            proven: [B] boolean - whether goal is provable (only valid if found)
            depth: [B] int - proof depth (only valid if found and proven)
        """
        B = goals.shape[0]
        device = self.device

        if B == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        # Hash to get initial indices
        idx = self._hash(goals)  # [B]

        # Linear probing with fixed max probes for compilation
        MAX_PROBES = 16
        found = torch.zeros(B, dtype=torch.bool, device=device)
        proven = torch.zeros(B, dtype=torch.bool, device=device)
        depth = torch.full((B,), -1, dtype=torch.long, device=device)

        for probe in range(MAX_PROBES):
            probe_idx = (idx + probe) % self.capacity

            # Check if slot is occupied
            slot_occupied = self.occupied[probe_idx]  # [B]

            # Get stored entry
            entry = self.table[probe_idx]  # [B, 5]
            stored_goal = entry[:, :3]  # [B, 3]
            stored_proven = entry[:, 3]  # [B]
            stored_depth = entry[:, 4]  # [B]

            # Match if occupied AND goals match
            match = slot_occupied & (stored_goal == goals).all(dim=1)

            # Update results for matches (only if not already found)
            newly_found = match & ~found
            found = found | match
            proven = torch.where(newly_found, stored_proven.bool(), proven)
            depth = torch.where(newly_found, stored_depth, depth)

            # Early exit check (can't use .all() for compilation, so continue)

        return found, proven, depth

    @torch.no_grad()
    def insert(self, goals: Tensor, proven: Tensor, depths: Tensor) -> None:
        """
        Insert proof results into table.

        Args:
            goals: [B, 3] ground atoms
            proven: [B] boolean - whether provable
            depths: [B] int - proof depth (-1 if not proven)
        """
        B = goals.shape[0]
        if B == 0:
            return

        # Hash to get initial indices
        idx = self._hash(goals)  # [B]

        # Linear probing for insertion
        MAX_PROBES = 16
        inserted = torch.zeros(B, dtype=torch.bool, device=self.device)

        for probe in range(MAX_PROBES):
            probe_idx = (idx + probe) % self.capacity

            # Check for empty slot or existing entry with same key
            slot_occupied = self.occupied[probe_idx]  # [B]
            entry = self.table[probe_idx]
            stored_goal = entry[:, :3]
            same_key = slot_occupied & (stored_goal == goals).all(dim=1)

            # Can insert if: empty slot OR same key (update)
            can_insert = (~slot_occupied | same_key) & ~inserted

            # Build new entries
            new_entry = torch.stack([
                goals[:, 0],
                goals[:, 1],
                goals[:, 2],
                proven.long(),
                depths.long(),
            ], dim=1)  # [B, 5]

            # Scatter update for entries that can be inserted
            insert_mask = can_insert
            if insert_mask.any():
                insert_indices = probe_idx[insert_mask]
                self.table[insert_indices] = new_entry[insert_mask]
                self.occupied[insert_indices] = True
                inserted = inserted | can_insert

        self.inserts += int(inserted.sum().item())

    def clear(self) -> None:
        """Clear all entries from the table."""
        self.table.zero_()
        self.occupied.zero_()
        self.hits = 0
        self.misses = 0
        self.inserts = 0

    def stats(self) -> dict:
        """Return table statistics."""
        return {
            "capacity": self.capacity,
            "occupied": int(self.occupied.sum().item()),
            "load_factor": float(self.occupied.sum().item()) / self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "inserts": self.inserts,
        }


class ProofTableBatch:
    """
    Batch-aware proof table that maintains separate caches per batch element.

    This is useful when different batch elements represent different queries
    that should not share cached results.
    """

    def __init__(
        self,
        batch_size: int,
        capacity_per_batch: int,
        device: torch.device,
        pack_base: int = 100000,
    ):
        """
        Initialize batch proof table.

        Args:
            batch_size: Number of batch elements
            capacity_per_batch: Cache capacity per batch element
            device: Torch device
            pack_base: Hashing base
        """
        self.batch_size = batch_size
        self.capacity = capacity_per_batch
        self.device = device
        self.pack_base = pack_base

        # Table: [batch_size, capacity, 5]
        self.table = torch.zeros(
            batch_size, capacity_per_batch, 5,
            dtype=torch.long, device=device
        )
        self.occupied = torch.zeros(
            batch_size, capacity_per_batch,
            dtype=torch.bool, device=device
        )

        if hasattr(torch, "_dynamo"):
            torch._dynamo.mark_static_address(self.table)
            torch._dynamo.mark_static_address(self.occupied)

    def _hash(self, goals: Tensor) -> Tensor:
        """Hash goals per batch. goals: [B, 3] -> indices: [B]"""
        p, a, b = goals[:, 0].long(), goals[:, 1].long(), goals[:, 2].long()
        h = ((p * self.pack_base) + a) * self.pack_base + b
        return h.abs() % self.capacity

    @torch.no_grad()
    def lookup(
        self,
        batch_idx: Tensor,  # [N] which batch element
        goals: Tensor,      # [N, 3] goals to lookup
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Look up proof results.

        Args:
            batch_idx: [N] batch indices (0 to batch_size-1)
            goals: [N, 3] ground atoms

        Returns:
            found: [N] boolean
            proven: [N] boolean
            depth: [N] int
        """
        N = goals.shape[0]
        device = self.device

        if N == 0:
            return (
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        idx = self._hash(goals)  # [N]

        MAX_PROBES = 16
        found = torch.zeros(N, dtype=torch.bool, device=device)
        proven = torch.zeros(N, dtype=torch.bool, device=device)
        depth = torch.full((N,), -1, dtype=torch.long, device=device)

        for probe in range(MAX_PROBES):
            probe_idx = (idx + probe) % self.capacity

            # Gather from per-batch tables
            slot_occupied = self.occupied[batch_idx, probe_idx]  # [N]
            entry = self.table[batch_idx, probe_idx]  # [N, 5]
            stored_goal = entry[:, :3]
            stored_proven = entry[:, 3]
            stored_depth = entry[:, 4]

            match = slot_occupied & (stored_goal == goals).all(dim=1)
            newly_found = match & ~found
            found = found | match
            proven = torch.where(newly_found, stored_proven.bool(), proven)
            depth = torch.where(newly_found, stored_depth, depth)

        return found, proven, depth

    @torch.no_grad()
    def insert(
        self,
        batch_idx: Tensor,  # [N]
        goals: Tensor,      # [N, 3]
        proven: Tensor,     # [N]
        depths: Tensor,     # [N]
    ) -> None:
        """Insert proof results into batch-specific caches."""
        N = goals.shape[0]
        if N == 0:
            return

        idx = self._hash(goals)
        MAX_PROBES = 16
        inserted = torch.zeros(N, dtype=torch.bool, device=self.device)

        for probe in range(MAX_PROBES):
            probe_idx = (idx + probe) % self.capacity

            slot_occupied = self.occupied[batch_idx, probe_idx]
            entry = self.table[batch_idx, probe_idx]
            stored_goal = entry[:, :3]
            same_key = slot_occupied & (stored_goal == goals).all(dim=1)

            can_insert = (~slot_occupied | same_key) & ~inserted

            if can_insert.any():
                new_entry = torch.stack([
                    goals[:, 0],
                    goals[:, 1],
                    goals[:, 2],
                    proven.long(),
                    depths.long(),
                ], dim=1)

                insert_indices = torch.where(can_insert)[0]
                for i in insert_indices:
                    bi = batch_idx[i].item()
                    pi = probe_idx[i].item()
                    self.table[bi, pi] = new_entry[i]
                    self.occupied[bi, pi] = True

                inserted = inserted | can_insert

    def clear(self, batch_indices: Optional[Tensor] = None) -> None:
        """Clear table entries. If batch_indices provided, only clear those."""
        if batch_indices is None:
            self.table.zero_()
            self.occupied.zero_()
        else:
            self.table[batch_indices].zero_()
            self.occupied[batch_indices].zero_()
