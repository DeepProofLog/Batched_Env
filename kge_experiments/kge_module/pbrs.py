"""
KGE-based Potential-Based Reward Shaping (PBRS) Module.

This module provides potential-based reward shaping using KGE scores.
Formula: r' = r + γ*Φ(s') - Φ(s), where Φ(s) = β * log(KGE_score(first_goal))

Two modes:
- Precomputed: Load potentials from pre-computed top-k files (fast)
- Runtime: Score atoms at runtime via KGE engine (slower but handles any state)

Reference: SB3 implementation in sb3_env.py:246-281
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


class PBRSModule:
    """Potential-Based Reward Shaping using KGE scores.

    Attributes:
        beta: PBRS weight (Φ(s) = beta * log(kge_score))
        gamma: Discount factor for shaping
        mode: 'precompute' (use cached potentials) or 'runtime' (score on-the-fly)
        device: Target device for tensors
    """

    def __init__(
        self,
        beta: float = 0.1,
        gamma: float = 0.99,
        mode: str = 'precompute',
        device: Optional[torch.device] = None,
        kge_engine: Optional[Any] = None,
        index_manager: Optional[Any] = None,
        eps: float = 1e-9,
    ) -> None:
        """Initialize PBRS module.

        Args:
            beta: PBRS weight. Φ(s) = beta * log(kge_score). 0 disables shaping.
            gamma: Discount factor for reward shaping.
            mode: 'precompute' or 'runtime'.
            device: Target device.
            kge_engine: KGE inference engine (required for runtime mode).
            index_manager: Index manager for atom conversion (required for runtime mode).
            eps: Small constant for numerical stability.
        """
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.device = device or torch.device('cpu')
        self.kge_engine = kge_engine
        self.index_manager = index_manager
        self.eps = eps

        # Potential cache: atom_key -> potential value
        self._potential_cache: Dict[str, float] = {}

        # For precomputed mode: tensor-based lookup
        # Maps packed atom hash -> potential
        self._precomputed_potentials: Optional[Dict[int, float]] = None

        # Terminal predicates (potential = 0)
        self._terminal_predicates = {'True', 'False', 'End', 'Endf', 'Endt'}

    def set_terminal_predicates(self, predicates: set) -> None:
        """Set terminal predicates (these get potential = 0)."""
        self._terminal_predicates = predicates

    def load_precomputed_potentials(self, filepath: str, pack_base: int) -> None:
        """Load pre-computed potentials from a top-k scores file.

        Args:
            filepath: Path to top-k file (format: "predicate(head,tail) score rank")
            pack_base: Base for packing atom indices into hash key.
        """
        self._precomputed_potentials = {}

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                atom_repr = parts[0]
                try:
                    score = float(parts[1])
                except ValueError:
                    continue

                # Compute potential: beta * log(score)
                score = max(score, self.eps)
                potential = self.beta * math.log(score)

                # Store by atom string (for now, could convert to hash later)
                self._potential_cache[atom_repr] = potential

        print(f"[PBRS] Loaded {len(self._potential_cache)} precomputed potentials")

    def compute_potential_batch(
        self,
        states: Tensor,  # [B, A, 3] current states
        idx2pred: Dict[int, str],
        idx2const: Dict[int, str],
        padding_idx: int = 0,
    ) -> Tensor:
        """Compute potentials for a batch of states.

        Args:
            states: [B, A, 3] state tensor (predicate, arg0, arg1).
            idx2pred: Index to predicate name mapping.
            idx2const: Index to constant name mapping.
            padding_idx: Padding index.

        Returns:
            [B] tensor of potentials.
        """
        B = states.shape[0]
        potentials = torch.zeros(B, dtype=torch.float32, device=self.device)

        if self.beta == 0.0:
            return potentials

        # Get first non-padding atom for each state
        first_preds = states[:, 0, 0].cpu().tolist()

        for b in range(B):
            pred_idx = first_preds[b]
            if pred_idx == padding_idx:
                continue

            pred_name = idx2pred.get(pred_idx)
            if pred_name is None:
                continue

            # Terminal states get 0 potential
            if pred_name in self._terminal_predicates:
                continue

            # Get arguments
            arg0_idx = states[b, 0, 1].item()
            arg1_idx = states[b, 0, 2].item()
            arg0 = idx2const.get(arg0_idx, str(arg0_idx))
            arg1 = idx2const.get(arg1_idx, str(arg1_idx))

            atom_str = f"{pred_name}({arg0},{arg1})"

            # Check cache first
            if atom_str in self._potential_cache:
                potentials[b] = self._potential_cache[atom_str]
            elif self.mode == 'runtime' and self.kge_engine is not None:
                # Score at runtime
                try:
                    scores = self.kge_engine.predict_batch([atom_str])
                    if scores:
                        score = max(float(scores[0]), self.eps)
                        potential = self.beta * math.log(score)
                        self._potential_cache[atom_str] = potential
                        potentials[b] = potential
                except Exception:
                    pass

        return potentials

    def compute_shaped_rewards(
        self,
        rewards: Tensor,  # [B] original rewards
        phi_s: Tensor,  # [B] potential of current states
        phi_sp: Tensor,  # [B] potential of next states
        done_mask: Tensor,  # [B] done flags
    ) -> Tensor:
        """Compute shaped rewards using PBRS formula.

        Args:
            rewards: [B] original rewards.
            phi_s: [B] potentials of current states.
            phi_sp: [B] potentials of next states.
            done_mask: [B] boolean done flags (terminal states get Φ(s')=0).

        Returns:
            [B] shaped rewards: r' = r + γ*Φ(s') - Φ(s)
        """
        # Terminal states get 0 potential
        phi_sp_effective = torch.where(
            done_mask,
            torch.zeros_like(phi_sp),
            phi_sp,
        )

        # PBRS formula: r' = r + γ*Φ(s') - Φ(s)
        shaped = rewards + self.gamma * phi_sp_effective - phi_s

        return shaped

    def clear_cache(self) -> None:
        """Clear the potential cache."""
        self._potential_cache.clear()

    @property
    def cache_size(self) -> int:
        """Return number of cached potentials."""
        return len(self._potential_cache)


class PBRSWrapper:
    """Wrapper that applies PBRS to a PPO algorithm.

    This wrapper intercepts rewards from the environment and applies
    potential-based reward shaping before passing them to PPO.

    Usage:
        pbrs = PBRSWrapper(ppo, pbrs_module, index_manager)
        # During rollout:
        shaped_reward = pbrs.shape_reward(reward, current_state, next_state, done)
    """

    def __init__(
        self,
        pbrs_module: PBRSModule,
        index_manager: Any,
    ) -> None:
        """Initialize PBRS wrapper.

        Args:
            pbrs_module: PBRS module instance.
            index_manager: Index manager for atom conversion.
        """
        self.pbrs = pbrs_module
        self.im = index_manager

        # Build reverse lookups
        self.idx2pred = {v: k for k, v in index_manager.predicate_str2idx.items()}
        self.idx2const = {v: k for k, v in index_manager.constant_str2idx.items()}
        self.padding_idx = index_manager.padding_idx

        # Cache last potential for each environment
        self._last_phi: Optional[Tensor] = None

    def reset(self, initial_states: Tensor) -> None:
        """Reset PBRS state for new episodes.

        Args:
            initial_states: [B, A, 3] initial state tensor.
        """
        self._last_phi = self.pbrs.compute_potential_batch(
            initial_states,
            self.idx2pred,
            self.idx2const,
            self.padding_idx,
        )

    def shape_rewards(
        self,
        rewards: Tensor,
        next_states: Tensor,
        done_mask: Tensor,
        reset_mask: Optional[Tensor] = None,
        reset_states: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply PBRS to rewards.

        Args:
            rewards: [B] original rewards.
            next_states: [B, A, 3] next state tensor.
            done_mask: [B] done flags.
            reset_mask: [B] which envs were reset this step.
            reset_states: [B, A, 3] new initial states for reset envs.

        Returns:
            [B] shaped rewards.
        """
        if self._last_phi is None:
            # First call, just compute next potentials
            self._last_phi = self.pbrs.compute_potential_batch(
                next_states,
                self.idx2pred,
                self.idx2const,
                self.padding_idx,
            )
            return rewards

        phi_s = self._last_phi

        # Compute next state potentials
        phi_sp = self.pbrs.compute_potential_batch(
            next_states,
            self.idx2pred,
            self.idx2const,
            self.padding_idx,
        )

        # Apply PBRS formula
        shaped = self.pbrs.compute_shaped_rewards(rewards, phi_s, phi_sp, done_mask)

        # Update last phi for next step
        if reset_mask is not None and reset_states is not None:
            # For reset envs, use potential of new initial state
            reset_phi = self.pbrs.compute_potential_batch(
                reset_states,
                self.idx2pred,
                self.idx2const,
                self.padding_idx,
            )
            # reset_mask is [B], reset_phi and phi_sp are [B] - no unsqueeze needed
            self._last_phi = torch.where(reset_mask, reset_phi, phi_sp)
        else:
            self._last_phi = phi_sp

        return shaped


def create_pbrs_module(
    config: Any,
    kge_engine: Optional[Any] = None,
    index_manager: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Optional[PBRSModule]:
    """Factory function to create PBRS module from config.

    Args:
        config: TrainConfig with PBRS settings.
        kge_engine: KGE inference engine.
        index_manager: Index manager.
        device: Target device.

    Returns:
        PBRSModule if enabled (beta != 0), None otherwise.
    """
    beta = getattr(config, 'pbrs_beta', 0.0)
    if beta == 0.0:
        return None

    gamma = getattr(config, 'pbrs_gamma', 0.99)
    precompute = getattr(config, 'pbrs_precompute', True)
    mode = 'precompute' if precompute else 'runtime'

    pbrs = PBRSModule(
        beta=beta,
        gamma=gamma,
        mode=mode,
        device=device,
        kge_engine=kge_engine,
        index_manager=index_manager,
    )

    # If precompute mode and we have top-k files, load them
    if mode == 'precompute':
        import os
        dataset = getattr(config, 'dataset', '')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        topk_path = os.path.join(base_dir, 'top_k_scores', 'files', f'kge_top_{dataset}.txt')

        if os.path.exists(topk_path):
            pack_base = getattr(index_manager, 'pack_base', 10000) if index_manager else 10000
            pbrs.load_precomputed_potentials(topk_path, pack_base)
        else:
            print(f"[PBRS] Warning: No precomputed scores at {topk_path}, using runtime mode")
            pbrs.mode = 'runtime'

    return pbrs
