"""
Ensemble KGE Models Module.

Combines scores from multiple KGE architectures (RotatE, ComplEx, TransE, etc.)
for more robust ranking from diverse embedding spaces.

Purpose: Different KGE models capture different relational patterns.
Ensembling provides more robust predictions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class KGEEnsemble(nn.Module):
    """Ensemble of multiple KGE models.

    Combines predictions from multiple KGE models using:
    - 'mean': Average scores across models
    - 'max': Take maximum score
    - 'learned': Learned weighted combination

    Attributes:
        models: List of KGE inference engines.
        method: Combination method ('mean', 'max', 'learned').
        weights: Learned weights for 'learned' method.
    """

    def __init__(
        self,
        models: List[Any],
        method: str = 'mean',
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize ensemble.

        Args:
            models: List of KGE inference engines.
            method: Combination method ('mean', 'max', 'learned').
            device: Target device.
        """
        super().__init__()
        self.models = models
        self.method = method
        self.n_models = len(models)

        if method == 'learned':
            # Initialize uniform weights
            init_weights = torch.ones(self.n_models) / self.n_models
            self.weights = nn.Parameter(init_weights)
        else:
            self.register_buffer('weights', None)

        if device is not None:
            self.to(device)

    @property
    def effective_weights(self) -> Tensor:
        """Return normalized weights for learned method."""
        if self.method == 'learned':
            return torch.softmax(self.weights, dim=0)
        else:
            return torch.ones(self.n_models) / self.n_models

    def predict_batch(self, atoms: Sequence[str]) -> Tensor:
        """Score atoms using ensemble of models.

        Args:
            atoms: List of atom strings to score.

        Returns:
            [N] ensemble scores for each atom.
        """
        # Collect scores from all models
        all_scores = []
        for model in self.models:
            scores = model.predict_batch(atoms)
            all_scores.append(scores)

        # Stack scores: [N, M] where M = number of models
        stacked = torch.stack(all_scores, dim=-1)

        # Combine based on method
        if self.method == 'mean':
            return stacked.mean(dim=-1)
        elif self.method == 'max':
            return stacked.max(dim=-1)[0]
        elif self.method == 'learned':
            weights = self.effective_weights.to(stacked.device)
            return (stacked * weights).sum(dim=-1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def forward(self, atoms: Sequence[str]) -> Tensor:
        """Alias for predict_batch."""
        return self.predict_batch(atoms)

    def __repr__(self) -> str:
        weights_str = ""
        if self.method == 'learned':
            w = self.effective_weights.detach().cpu().numpy()
            weights_str = f", weights={w}"
        return f"KGEEnsemble(n_models={self.n_models}, method={self.method}{weights_str})"


class EnsembleBridge(nn.Module):
    """Neural bridge for ensemble: combines RL + multiple KGE scores.

    Formula: score = alpha_rl * rl + sum(alpha_kge[i] * kge[i])

    Attributes:
        alpha_rl: Learnable weight for RL.
        alpha_kge: Learnable weights for each KGE model.
    """

    def __init__(
        self,
        n_kge_models: int,
        init_alpha_rl: float = 0.3,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize ensemble bridge.

        Args:
            n_kge_models: Number of KGE models in ensemble.
            init_alpha_rl: Initial weight for RL (KGE weights sum to 1-alpha_rl).
            device: Target device.
        """
        super().__init__()
        self.n_kge_models = n_kge_models

        # Initialize weights (unconstrained, will be softmaxed)
        # [alpha_rl, alpha_kge_1, ..., alpha_kge_n]
        n_total = 1 + n_kge_models
        init_weights = torch.zeros(n_total)
        init_weights[0] = torch.logit(torch.tensor(init_alpha_rl))
        self.weights = nn.Parameter(init_weights)

        if device is not None:
            self.to(device)

    @property
    def effective_weights(self) -> Tensor:
        """Return normalized weights."""
        return torch.softmax(self.weights, dim=0)

    @property
    def effective_alpha_rl(self) -> float:
        """Return effective RL weight."""
        return self.effective_weights[0].item()

    def forward(
        self,
        rl_logprobs: Tensor,  # [B, K]
        kge_logprobs_list: List[Tensor],  # List of [B, K]
        success_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Combine RL and multiple KGE scores.

        Args:
            rl_logprobs: [B, K] RL log probabilities.
            kge_logprobs_list: List of [B, K] KGE log scores per model.
            success_mask: Optional [B, K] proof success mask (unused).

        Returns:
            [B, K] combined scores.
        """
        weights = self.effective_weights

        # RL contribution
        score = weights[0] * rl_logprobs

        # KGE contributions
        for i, kge_logprobs in enumerate(kge_logprobs_list):
            score = score + weights[1 + i] * kge_logprobs

        return score

    def __repr__(self) -> str:
        w = self.effective_weights.detach().cpu().numpy()
        return f"EnsembleBridge(alpha_rl={w[0]:.4f}, alpha_kge={w[1:]})"


def load_ensemble_models(
    signatures: str,
    config: Any,
    verbose: bool = True,
) -> List[Any]:
    """Load multiple KGE models for ensemble.

    Args:
        signatures: Comma-separated run signatures.
        config: TrainConfig with KGE settings.
        verbose: Print loading information.

    Returns:
        List of KGE inference engines.
    """
    # Import here to avoid circular imports
    from kge_module.inference import KGEInference

    signature_list = [s.strip() for s in signatures.split(',')]
    models = []

    checkpoint_dir = getattr(config, 'kge_checkpoint_dir', None)
    engine = getattr(config, 'kge_engine', 'pytorch')

    for sig in signature_list:
        if verbose:
            print(f"[KGEEnsemble] Loading model: {sig}")

        try:
            model = KGEInference(
                checkpoint_dir=checkpoint_dir,
                run_signature=sig,
                engine=engine,
            )
            models.append(model)
        except Exception as e:
            print(f"[KGEEnsemble] Failed to load {sig}: {e}")

    if verbose:
        print(f"[KGEEnsemble] Loaded {len(models)} models")

    return models


def create_kge_ensemble(
    config: Any,
    device: Optional[torch.device] = None,
) -> Optional[KGEEnsemble]:
    """Factory function to create KGE ensemble from config.

    Args:
        config: TrainConfig with ensemble settings.
        device: Target device.

    Returns:
        KGEEnsemble if enabled, None otherwise.
    """
    if not getattr(config, 'kge_ensemble', False):
        return None

    signatures = getattr(config, 'kge_ensemble_signatures', None)
    if not signatures:
        print("[KGEEnsemble] No signatures provided, skipping ensemble")
        return None

    method = getattr(config, 'kge_ensemble_method', 'mean')
    verbose = getattr(config, 'verbose', True)

    models = load_ensemble_models(signatures, config, verbose)
    if not models:
        print("[KGEEnsemble] No models loaded, skipping ensemble")
        return None

    return KGEEnsemble(models=models, method=method, device=device)


def create_ensemble_bridge(
    config: Any,
    n_kge_models: int,
    device: Optional[torch.device] = None,
) -> Optional[EnsembleBridge]:
    """Factory function to create ensemble bridge from config.

    Args:
        config: TrainConfig with bridge settings.
        n_kge_models: Number of KGE models in ensemble.
        device: Target device.

    Returns:
        EnsembleBridge if enabled, None otherwise.
    """
    if not getattr(config, 'kge_ensemble', False):
        return None

    if n_kge_models < 1:
        return None

    init_alpha_rl = getattr(config, 'neural_bridge_init_alpha', 0.3)
    verbose = getattr(config, 'verbose', True)

    bridge = EnsembleBridge(
        n_kge_models=n_kge_models,
        init_alpha_rl=init_alpha_rl,
        device=device,
    )

    if verbose:
        print(f"[EnsembleBridge] Created with {n_kge_models} KGE models, "
              f"init_alpha_rl={init_alpha_rl}")

    return bridge
