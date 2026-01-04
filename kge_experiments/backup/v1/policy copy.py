"""
Production-ready Policy and Value Networks for Actor-Critic RL.

CONVENTIONS:
    B: Batch size (number of parallel environments)
    S: Action space size (number of successor states)
    E: Embedding dimension 
    H: Hidden dimension (MLP width)
    A: Number of atoms (aggregated before reaching the network)

Standard Flow:
    1. CustomCombinedExtractor: Indices [B, S, A, 3] -> Embeddings [B, S, E]
    2. SharedPolicyValueNetwork: Embeddings [B, S, E] -> Logits [B, S] & Values [B]
    3. ActorCriticPolicy: Orchestrates extraction, forward pass, and action distribution
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional, Dict, Union
from tensordict import TensorDict
import os

# Use FastCategorical if available for better performance in compiled mode
if os.environ.get("USE_FAST_CATEGORICAL", "1") == "1":
    try:
        from utils.distributions import FastCategoricalDistribution as CategoricalDistribution
    except ImportError:
        from stable_baselines3.common.distributions import CategoricalDistribution
else:
    from stable_baselines3.common.distributions import CategoricalDistribution

from stable_baselines3.common.policies import BasePolicy

# Handle both relative and absolute imports
try:
    from .kernels import FusedLinearReluLayerNorm, FusedLinearRelu
except ImportError:
    from kernels import FusedLinearReluLayerNorm, FusedLinearRelu


class CustomCombinedExtractor(nn.Module):
    """
    Feature extractor converting index-based observations [B, S, A, 3] into dense embeddings [B, S, E].
    """
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.embed_dim = embedder.embedding_dim
        self._features_dim = 1 # Required for SB3 scaffold compatibility

    def forward(self, obs: Union[dict, TensorDict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: Dictionary containing 'sub_index' [B, 1, A, 3], 'derived_sub_indices' [B, S, A, 3], 'action_mask' [B, S]
        Returns:
            obs_embeddings: [B, 1, E]
            action_embeddings: [B, S, E]
            action_mask: [B, S]
        """
        # Get tensors from dict/TensorDict
        if isinstance(obs, (dict, TensorDict)):
            obs_sub_indices = obs.get("sub_index")
            action_sub_indices = obs.get("derived_sub_indices")
            action_mask = obs.get("action_mask")
        else:
            # Fallback for old-style positional arguments if needed (not recommended)
            obs_sub_indices, action_sub_indices, action_mask = obs
        
        # Note: obs tensors should already be int32/long; casting removed for CUDAGraph compatibility
        
        # Aggregate atoms internally: [B, S, A, 3] -> [B, S, E]
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices) # [B, 1, E]
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices) # [B, S, E]
        
        return obs_embeddings, action_embeddings, action_mask


class SharedBody(nn.Module):
    """
    Residual MLP backbone [E] -> [H] -> [H] shared by policy and value heads.
    Uses fused Linear+ReLU+LayerNorm modules for better performance.
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 8, parity: bool = False):
        super().__init__()
        self.parity = parity

        if parity:
            # Parity mode: Use Sequential matching tensor policy exactly (Linear, ReLU, LayerNorm, Dropout)
            self.input_transform = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.0)
            )
            self.res_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.0)
                ) for _ in range(num_layers)
            ])
        else:
            # Optimized mode: fused Linear+ReLU+LayerNorm
            self.input_transform = FusedLinearReluLayerNorm(embed_dim, hidden_dim)
            self.res_blocks = nn.ModuleList([
                FusedLinearReluLayerNorm(hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [..., E]
        Returns:
            Features [..., H]
        """
        if self.parity:
            # Parity mode: Sequential contains Linear, ReLU, LayerNorm, Dropout
            x = self.input_transform(x)
            for block in self.res_blocks:
                identity = x
                x = block(x)  # Sequential: Linear, ReLU, LayerNorm, Dropout
                x = x + identity  # Residual
            return x
        else:
            # Optimized mode
            x = self.input_transform(x)
            for block in self.res_blocks:
                x = x + block(x)
            return x


class SharedPolicyValueNetwork(nn.Module):
    """
    Unified network architecture for policy and value estimation.
    Merges former CustomNetwork and SharedPolicyValueNetwork functionality.
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 8,
                 temperature: Optional[float] = None, use_l2_norm: bool = False, sqrt_scale: bool = True, parity: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.use_l2_norm = use_l2_norm
        self.sqrt_scale = sqrt_scale
        self.parity = parity

        self.shared_body = SharedBody(embed_dim, hidden_dim, num_layers, parity=parity)

        if parity:
            # Parity mode: Use Sequential to match tensor policy structure
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            # Optimized mode: fused layers
            # Policy head: Projects shared features back to embedding space for dot-product attention
            # First layer: fused Linear+ReLU, second layer: plain Linear (no activation after)
            self.policy_head_fused = FusedLinearRelu(hidden_dim, hidden_dim)
            self.policy_head_final = nn.Linear(hidden_dim, embed_dim)

            # Value head: Projects shared features to a scalar state-value estimate
            # First layer: fused Linear+ReLU, second layer: plain Linear (no activation after)
            self.value_head_fused = FusedLinearRelu(hidden_dim, hidden_dim // 2)
            self.value_head_final = nn.Linear(hidden_dim // 2, 1)

        # SB3 scaffold compatibility
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

    def _get_shared_features(self, embeddings: torch.Tensor) -> torch.Tensor:
        """[..., E] -> [..., H]"""
        return self.shared_body(embeddings)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint forward pass for training (Actor + Critic).
        Args:
            features: (obs_embeddings [B, 1, E], act_embeddings [B, S, E], act_mask [B, S])
        Returns:
            logits: [B, S]
            values: [B]
        """
        obs_emb, act_emb, act_mask = features
        shared_obs = self._get_shared_features(obs_emb) # [B, 1, H]
        shared_act = self._get_shared_features(act_emb) # [B, S, H]

        # Compute Logits
        if self.parity:
            p_obs = self.policy_head(shared_obs) # [B, 1, E]
            p_act = self.policy_head(shared_act) # [B, S, E]
        else:
            p_obs = self.policy_head_fused(shared_obs) # [B, 1, H]
            p_obs = self.policy_head_final(p_obs) # [B, 1, E]
            p_act = self.policy_head_fused(shared_act) # [B, S, H]
            p_act = self.policy_head_final(p_act) # [B, S, E]

        if self.use_l2_norm:
            p_obs = F.normalize(p_obs, dim=-1)
            p_act = F.normalize(p_act, dim=-1)

        logits = torch.matmul(p_obs, p_act.transpose(-2, -1)).squeeze(-2) # [B, S]

        if self.sqrt_scale:
            logits = logits / (self.embed_dim ** 0.5)
        if self.temperature is not None:
            logits = logits / self.temperature

        logits = logits.masked_fill(~act_mask.bool(), float("-inf"))

        # Compute Values
        if self.parity:
            values = self.value_head(shared_obs).squeeze(-1).squeeze(-1) # [B]
        else:
            values = self.value_head_fused(shared_obs) # [B, 1, H//2]
            values = self.value_head_final(values).squeeze(-1).squeeze(-1) # [B]
        
        return logits, values

    def forward_actor(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Actor-only pass [B, S]"""
        logits, _ = self.forward(features)
        return logits

    def forward_critic(self, obs_embeddings: torch.Tensor) -> torch.Tensor:
        """Critic-only pass [B]"""
        shared_obs = self._get_shared_features(obs_embeddings)
        if self.parity:
            return self.value_head(shared_obs).squeeze(-1).squeeze(-1)
        else:
            values = self.value_head_fused(shared_obs)
            return self.value_head_final(values).squeeze(-1).squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """
    Main Policy class orchestrating extraction, architecture, and action distribution.
    """
    def __init__(self, embedder, embed_dim: int, hidden_dim: int, num_layers: int, 
                 device: torch.device, action_dim: int = None, parity: bool = False, **kwargs):
        super().__init__()
        self.device = device
        self.features_extractor = CustomCombinedExtractor(embedder)
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        
        self.mlp_extractor = SharedPolicyValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            temperature=kwargs.get('temperature'),
            use_l2_norm=kwargs.get('use_l2_norm', False),
            sqrt_scale=kwargs.get('sqrt_scale', True),
            parity=parity
        )
        
        # Action distribution
        self.action_dist = CategoricalDistribution(action_dim)
        
        # SB3 Parity Scaffold: Dummy layers to match RNG consumption during initialization
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, action_dim if action_dim else 1)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        
        # Initialization
        self._init_weights(parity)
        self.to(device)

    def _init_weights(self, parity: bool):
        """Standard SB3-style initialization sequence."""
        module_gains = {
            self.features_extractor: math.sqrt(2.0),
            self.mlp_extractor: math.sqrt(2.0),
            self.action_net: 0.01,
            self.value_net: 1.0,
        }
        for module, gain in module_gains.items():
            module.apply(partial(BasePolicy.init_weights, gain=gain))
            
        if not parity:
            # Refine value head for better initial stability
            self.mlp_extractor.value_head_fused.apply(partial(BasePolicy.init_weights, gain=1.0))
            self.mlp_extractor.value_head_final.apply(partial(BasePolicy.init_weights, gain=1.0))

    def forward(self, obs: Union[dict, TensorDict], deterministic: bool = False, 
                actions: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Unified forward pass.
        If actions=None: return (sampled_actions, values, log_probs) [B], [B], [B]
        If actions!=None: return (values, log_probs, entropy) [B], [B], [B]
        """
        features = self.features_extractor(obs)
        logits, values = self.mlp_extractor(features)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        if actions is not None:
            return values, distribution.log_prob(actions), distribution.entropy()
        
        actions = distribution.mode() if deterministic else distribution.sample()
        return actions, values, distribution.log_prob(actions)

    def evaluate_actions(self, sub_index: torch.Tensor, derived_sub_indices: torch.Tensor, 
                         action_mask: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized evaluation using raw tensors (Faster & CUDA graph stable).
        Args:
            sub_index: [B, 1, A, 3]
            derived_sub_indices: [B, S, A, 3]
            action_mask: [B, S]
            actions: [B]
        Returns:
            values [B], log_probs [B], entropy [B]
        """
        # Feature extraction (manual bypass of dict creation)
        obs_emb = self.features_extractor.embedder.get_embeddings_batch(sub_index.to(torch.int32))
        act_emb = self.features_extractor.embedder.get_embeddings_batch(derived_sub_indices.to(torch.int32))
        
        logits, values = self.mlp_extractor((obs_emb, act_emb, action_mask))
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        return values, distribution.log_prob(actions), distribution.entropy()

    def predict_values(self, obs: Union[dict, TensorDict]) -> torch.Tensor:
        """Value prediction [B]"""
        obs_emb, _, _ = self.features_extractor(obs)
        return self.mlp_extractor.forward_critic(obs_emb)

    def get_logits(self, obs: Union[dict, TensorDict]) -> torch.Tensor:
        """Evaluation-only: get raw action logits [B, S]"""
        features = self.features_extractor(obs)
        return self.mlp_extractor.forward_actor(features)


def create_actor_critic(embedder, embed_dim: int = 100, hidden_dim: int = 256, num_layers: int = 4,
                        device: torch.device = None, padding_states: int = None, **kwargs) -> ActorCriticPolicy:
    """Factory function for ActorCriticPolicy."""
    return ActorCriticPolicy(
        embedder=embedder, embed_dim=embed_dim, hidden_dim=hidden_dim, 
        num_layers=num_layers, device=device, action_dim=padding_states, **kwargs
    )
