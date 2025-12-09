"""
Neural Network Architecture for Actor-Critic Policy.

This module defines the neural network components for reinforcement learning:

    - CustomCombinedExtractor: Converts index-based observations to embeddings
    - SharedBody: Residual MLP backbone shared between policy and value heads
    - PolicyHead: Projects shared features to action embedding space
    - ValueHead: Produces scalar value estimates from shared features
    - SharedPolicyValueNetwork: Complete policy-value network with shared backbone
    - CustomNetwork: High-level network interface for the trainer
    - ActorCriticPolicy: Complete actor-critic policy with action sampling

Tensor Shape Conventions:
    B = Batch size (number of environments)
    S = Number of possible successor states (action space size)
    A = Number of atoms per state
    E = Embedding dimension
    H = Hidden dimension
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Tuple, Optional, Dict
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
import os
if os.environ.get("USE_FAST_CATEGORICAL", "1") == "1":
    try:
        from utils.fast_distributions import FastCategoricalDistribution as CategoricalDistribution
    except ImportError:
        from stable_baselines3.common.distributions import CategoricalDistribution
else:
    from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import MlpExtractor


class CustomCombinedExtractor(nn.Module):
    """
    Feature extractor that converts index-based observations into dense embeddings.
    
    This module takes observation indices (predicate, arg0, arg1) and produces
    continuous embedding vectors using a learned embedder.
    """
    
    def __init__(self, embedder):
        """Initialize the feature extractor.
        
        Args:
            embedder: Embedding module that converts indices to dense vectors.
        """
        super().__init__()
        self.embedder = embedder
        self.embed_dim = embedder.embedding_dim
        self._features_dim = 1
        self.features_dim = 1
    
    def forward(self, obs: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from observations.
        
        Args:
            obs: TensorDict containing:
                - 'sub_index': (batch, 1, pad_atoms, 3)
                - 'derived_sub_indices': (batch, pad_states, pad_atoms, 3)
                - 'action_mask': (batch, pad_states)
        
        Returns:
            obs_embeddings: (batch, 1, pad_atoms, embed_dim)
            action_embeddings: (batch, pad_states, pad_atoms, embed_dim)
            action_mask: (batch, pad_states)
        """
        # Get tensors from TensorDict
        obs_sub_indices = obs.get("sub_index")  # [B, 1, A, 3]
        action_sub_indices = obs.get("derived_sub_indices")  # [B, S, A, 3]
        action_mask = obs.get("action_mask")  # [B, S]
        
        # Ensure correct dtype
        if obs_sub_indices.dtype != torch.int32:
            obs_sub_indices = obs_sub_indices.to(torch.int32)  # [B, 1, A, 3]
        if action_sub_indices.dtype != torch.int32:
            action_sub_indices = action_sub_indices.to(torch.int32)  # [B, S, A, 3]
        
        # Get embeddings from embedder
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)  # [B, 1, A, E]
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)  # [B, S, A, E]
        
        # Extract valid_indices if present (e.g. from env optimization)
        valid_indices = obs.get("valid_indices", None)
        
        # If not provided by env, generate locally using CPU to avoid GPU-nonzero sync
        if valid_indices is None and action_mask is not None:
             # Detach and move to CPU to avoid GPU serialization bubble
             # Use persistent pinned memory buffer to speed up D2H transfer
             if not hasattr(self, '_pinned_mask_buffer'):
                 # Max size: 32 * 150 = 4800 (hardcoded for now as reasonable upper bound, or can resize dyn)
                 # We alloc slightly more to be safe
                 self._pinned_mask_buffer = torch.zeros(10000, dtype=torch.bool).pin_memory()
             
             flat_mask = action_mask.detach().view(-1)
             numel = flat_mask.numel()
             
             # Resize buffer if needed (rare)
             if numel > self._pinned_mask_buffer.numel():
                 self._pinned_mask_buffer = torch.zeros(numel * 2, dtype=torch.bool).pin_memory()
                 
             # Copy to pinned memory (async if standard)
             self._pinned_mask_buffer[:numel].copy_(flat_mask, non_blocking=True)
             
             # Sync happens here when we access the tensor content
             mask_cpu = self._pinned_mask_buffer[:numel]
             
             if mask_cpu.any():
                 # nonzero on CPU is fast
                 indices = torch.nonzero(mask_cpu).squeeze(-1)
                 # Move back to GPU (fast copy for small indices)
                 valid_indices = indices.to(action_mask.device)
             else:
                 valid_indices = torch.zeros(0, dtype=torch.long, device=action_mask.device)

        return obs_embeddings, action_embeddings, action_mask, valid_indices


class SharedBody(nn.Module):
    """
    Shared residual MLP backbone for both policy and value networks.
    
    The shared architecture enables efficient learning by allowing both
    policy and value heads to leverage common feature representations.
    Uses residual connections for stable gradient flow in deep networks.
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, num_layers: int = 8, dropout_prob: float = 0.0):
        """Initialize the shared body network.
        
        Args:
            embed_dim: Dimension of input embeddings.
            hidden_dim: Dimension of hidden layers in residual blocks.
            num_layers: Number of residual blocks.
            dropout_prob: Dropout probability (0.0 recommended for stable training).
        """
        super().__init__()
        
        # Initial projection from embedding to hidden dimension
        self.input_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Stack of residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        self.hidden_dim = hidden_dim

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None, valid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply shared residual network to embeddings.
        
        Args:
            embeddings: Input embeddings of shape [..., embed_dim].
            mask: Optional boolean mask of shape matching embeddings[:-1].
            valid_indices: Optional pre-calculated indices of valid elements [N_valid].
                  If provided, avoids 'nonzero' synchronization.
                  If provided, mask is ignored for index calculation but output relies on implicit shape.
            
        Returns:
            Processed features of shape [..., hidden_dim].
        """
        original_shape = embeddings.shape
        flat_embeddings = embeddings.view(-1, original_shape[-1])  # [N, embed_dim]
        
        # Optimization: Process only valid (masked) elements
        # If valid_indices provided (Strategy B - CPU Gen), use it to avoid sync.
        # Otherwise fallback to mask.
        indices = valid_indices
        
        if indices is None and mask is not None:
             # Fallback: compute indices from mask (incurs sync in eager)
             # We use nonzero here for robustness as it works with index_copy_
             indices = torch.nonzero(mask.reshape(-1)).squeeze(-1)
             
        if indices is not None:
            # Gather valid inputs
            # flat_embeddings is [N_total, E], indices is [N_valid]
            x_valid = flat_embeddings.index_select(0, indices)
            
            # Process valid inputs through network
            x_valid = self.input_transform(x_valid)
            for block in self.res_blocks:
                x_valid = block(x_valid) + x_valid
                
            # Scatter back to full shape
            output_flat = torch.zeros(flat_embeddings.shape[0], self.hidden_dim, 
                                    device=embeddings.device, dtype=embeddings.dtype)
            
            # Place processed values back
            output_flat.index_copy_(0, indices, x_valid)
            
            return output_flat.view(*original_shape[:-1], -1)
            
        else:
            # Fallback (slow path): Process everything
            x = self.input_transform(flat_embeddings)  # [N, hidden_dim]
            
            # Apply residual blocks
            for block in self.res_blocks:
                residual = x
                x = block(x) + residual  # [N, hidden_dim]
            
            return x.view(*original_shape[:-1], -1)  # [..., hidden_dim]


class PolicyHead(nn.Module):
    """
    Policy head that projects shared features to action embedding space.
    
    Used to compute action logits by comparing projected observation
    embeddings with projected action embeddings via dot product.
    """
    
    def __init__(self, hidden_dim: int = 256, embed_dim: int = 64):
        """Initialize the policy head.
        
        Args:
            hidden_dim: Dimension of input from shared body.
            embed_dim: Output embedding dimension for action comparison.
        """
        super().__init__()
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Project shared features to embedding space.
        
        Args:
            shared_features: Features from shared body [..., hidden_dim].
            
        Returns:
            Projected embeddings [..., embed_dim].
        """
        return self.out_transform(shared_features)


class ValueHead(nn.Module):
    """
    Value head that produces scalar value estimates from shared features.
    
    Estimates the expected cumulative reward from the current state.
    """
    
    def __init__(self, hidden_dim: int = 256):
        """Initialize the value head.
        
        Args:
            hidden_dim: Dimension of input from shared body.
        """
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from shared features.
        
        Args:
            shared_features: Features from shared body [..., hidden_dim].
            
        Returns:
            Scalar value estimates with last dimension squeezed [...].
        """
        value = self.output_layer(shared_features)  # [..., 1]
        return value.squeeze(-1)  # [...]


class SharedPolicyValueNetwork(nn.Module):
    """
    Combined policy-value network with shared backbone.
    
    Uses a shared residual network for feature extraction, with separate
    heads for policy (action selection) and value (state evaluation).
    
    The policy uses dot-product attention between observation and action
    embeddings to compute action logits.
    """    

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 256, 
                 num_layers: int = 8, dropout_prob: float = 0.0,
                 compile_model: bool = False,
                 use_amp: bool = False):
        """Initialize the policy-value network.
        
        Args:
            embed_dim: Dimension of input/output embeddings.
            hidden_dim: Hidden dimension in residual blocks.
            num_layers: Number of residual blocks in shared body.
            dropout_prob: Dropout probability (0.0 recommended).
            compile_model: Whether to apply torch.compile for speed.
            use_amp: Whether to use automatic mixed precision.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.shared_body = SharedBody(embed_dim, hidden_dim, num_layers, dropout_prob)
        self.policy_head = PolicyHead(hidden_dim, embed_dim)
        self.value_head = ValueHead(hidden_dim)
        self._use_amp = use_amp        
        # Apply torch.compile with bucketing strategy for maximum performance
        if compile_model:
            self.forward_policy = torch.compile(self.forward_policy)
            print(f"[Model] torch.compile applied to forward_policy")
            
    def _encode_with_shared_body(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None, valid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process embeddings through the shared backbone.
        
        Args:
            embeddings: Input embeddings [..., embed_dim].
            mask: Optional mask for embeddings.
            valid_indices: Optional valid indices.
            
        Returns:
            Processed features [..., hidden_dim].
        """
        return self.shared_body(embeddings, mask=mask, valid_indices=valid_indices)

    def forward_policy(self, obs_embeddings: torch.Tensor, 
                       action_embeddings: torch.Tensor, 
                       action_mask: torch.Tensor,
                       valid_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute action logits via observation-action similarity.
        
        Args:
            obs_embeddings: Observation embeddings [B, 1, A, E].
            action_embeddings: Action embeddings [B, S, A, E].
            action_mask: Boolean mask for valid actions [B, S].
            
        Returns:
            Action logits with invalid actions masked to -inf [B, S].
        """
        # Encode observations and actions through shared body + policy head
        shared_obs = self._encode_with_shared_body(obs_embeddings)  # [B, 1, A, H]
        encoded_obs = self.policy_head(shared_obs)  # [B, 1, A, E]
        # action_mask is [B, S], embeddings are [B, S, E] (aggregated over atoms)
        # So mask matches directly. NO expansion needed.
        
        shared_actions = self._encode_with_shared_body(action_embeddings, mask=action_mask, valid_indices=valid_indices)  # [B, S, H]
        encoded_actions = self.policy_head(shared_actions)  # [B, S, E]
        
        # Compute dot-product similarity for action selection
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)  # [B, S]
        
        # Scale logits by embedding dimension for stable gradients
        logits = logits / (obs_embeddings.shape[-1] ** 0.5)
        
        # Mask invalid actions
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))  # [B, S]
        
        return logits

    def forward_value(self, obs_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate.
        
        Args:
            obs_embeddings: Observation embeddings [B, 1, A, E].
            
        Returns:
            Value estimates [B].
        """
        shared_obs = self._encode_with_shared_body(obs_embeddings)  # [B, 1, A, H]
        value = self.value_head(shared_obs)  # [B]
        return value


class CustomNetwork(nn.Module):
    """
    High-level network interface combining policy and value networks.
    
    Wraps SharedPolicyValueNetwork to provide a clean interface for the
    PPO trainer and other training components.
    """
    
    def __init__(self, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 1, 
                 embed_dim: int = 200, 
                 compile_model: bool = False, 
                 use_amp: bool = False):
        """Initialize the custom network.
        
        Args:
            last_layer_dim_pi: Policy output dimension (unused, kept for compatibility).
            last_layer_dim_vf: Value output dimension (unused, kept for compatibility).
            embed_dim: Embedding dimension for the shared network.
            compile_model: Whether to apply torch.compile.
            use_amp: Whether to use AMP.
        """
        super().__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.shared_network = SharedPolicyValueNetwork(
            embed_dim, 
            compile_model=compile_model,
            use_amp=use_amp
        )

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both policy logits and value estimate.
        
        Args:
            features: Tuple of (obs_embeddings, action_embeddings, action_mask).
            
        Returns:
            Tuple of (action_logits [B, S], value_estimates [B]).
        """
        if len(features) == 4:
            obs_embeddings, action_embeddings, action_mask, valid_indices = features
            logits = self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask, valid_indices)
        else:
            obs_embeddings, action_embeddings, action_mask = features
            logits = self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask)
        value = self.shared_network.forward_value(obs_embeddings).squeeze(-1)  # [B]
        return logits, value
    
    def forward_actor(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                      ) -> torch.Tensor:
        """Compute action logits only.
        
        Args:
            features: Tuple of (obs_embeddings, action_embeddings, action_mask).
            
        Returns:
            Action logits [B, S].
        """
        if len(features) == 4:
            obs_embeddings, action_embeddings, action_mask, valid_indices = features
            return self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask, valid_indices)
        else:
             obs_embeddings, action_embeddings, action_mask = features
             return self.shared_network.forward_policy(obs_embeddings, action_embeddings, action_mask)

    def forward_critic(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                       ) -> torch.Tensor:
        """Compute value estimate only.
        
        Args:
            features: Tuple of (obs_embeddings, action_embeddings, action_mask).
            
        Returns:
            Value estimates [B].
        """
        obs_embeddings = features[0]
        return self.shared_network.forward_value(obs_embeddings).squeeze(-1)

    @property
    def value_network(self):
        """Return the value network computation function."""
        return self._value_network_wrapper

    def _value_network_wrapper(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute value from embeddings directly.
        
        Args:
            embeddings: Input embeddings [B, ...].
            
        Returns:
            Value estimates [B].
        """
        return self.shared_network.forward_value(embeddings)


class ActorCriticPolicy(nn.Module):
    """
    Complete actor-critic policy for PPO training.
    
    Combines feature extraction (embedder), shared network backbone,
    and action distribution to provide:
    - Action sampling with log probabilities
    - Action evaluation for PPO updates
    - Value estimation for advantage computation
    """
    
    def __init__(
        self,
        embedder,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_prob: float,
        device: torch.device,
        share_features_extractor: bool = True,
        init_seed: Optional[int] = None,
        action_dim: Optional[int] = None,
        **kwargs
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.device = device if device is not None else torch.device('cpu')
        self.share_features_extractor = share_features_extractor

        # Features extractor(s) – mirror SB3 CombinedExtractor contract and ordering
        self.features_extractor = CustomCombinedExtractor(embedder)
        if share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = CustomCombinedExtractor(embedder)

        # --- Base SB3-style scaffold (consumes RNG in the same order) ---
        # Build CustomNetwork as SB3 does inside _build_mlp_extractor (defaults)
        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=self.features_extractor._features_dim,
            last_layer_dim_vf=1,
            embed_dim=getattr(self.features_extractor, "embed_dim", embed_dim),
            compile_model=kwargs.get('compile_model', False),
            use_amp=kwargs.get('use_amp', False),
        )
        # Dummy action/value heads created during _build in SB3
        latent_pi = getattr(self.mlp_extractor, "latent_dim_pi", 1)
        latent_vf = getattr(self.mlp_extractor, "latent_dim_vf", 1)
        action_dim = int(action_dim) if action_dim is not None else -1
        self.action_net = nn.Linear(latent_pi, action_dim if action_dim > 0 else 1)
        self.value_net = nn.Linear(latent_vf, 1)

        # SB3-style orthogonal initialization sequence (match ActorCriticPolicy._build)
        module_gains = {
            self.features_extractor: math.sqrt(2.0),
            self.mlp_extractor: math.sqrt(2.0),
            self.action_net: 0.01,
            self.value_net: 1.0,
        }
        if not self.share_features_extractor:
            del module_gains[self.features_extractor]
            module_gains[self.pi_features_extractor] = math.sqrt(2.0)
            module_gains[self.vf_features_extractor] = math.sqrt(2.0)
        for module, gain in module_gains.items():
            module.apply(partial(BasePolicy.init_weights, gain=gain))

        # Action distribution - use fast version that skips Categorical validation
        self.action_dist = CategoricalDistribution(action_dim)

        # Move everything to device
        self.to(self.device)
    
    def extract_features(
        self, 
        obs: TensorDict,
        features_extractor: Optional[CustomCombinedExtractor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Extract features from observations - matches SB3 exactly.
        """
        if self.share_features_extractor:
            extractor = self.features_extractor if features_extractor is None else features_extractor
            return extractor(obs)
        else:
            # Extract features separately for policy and value
            pi_features = self.pi_features_extractor(obs)
            vf_features = self.vf_features_extractor(obs)
            return pi_features, vf_features
    
    def forward(
        self,
        obs: TensorDict,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - exactly matches SB3's forward.
        
        Args:
            obs: Observations as TensorDict
            deterministic: If True, select argmax action instead of sampling
        
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features following sb3 pattern
        features = self.extract_features(obs)
        if self.share_features_extractor:
            logits, values = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            logits = self.mlp_extractor.forward_actor(pi_features)
            values = self.mlp_extractor.forward_critic(vf_features)

        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Compute log probabilities
        action_log_probs = distribution.log_prob(actions)
        
        return actions, values, action_log_probs
    
    def evaluate_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to current policy - matches SB3 exactly.
        
        Args:
            obs: Observations as TensorDict
            actions: Actions to evaluate
        
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features
        features = self.extract_features(obs)

        if self.share_features_extractor:
            logits, values = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            logits = self.mlp_extractor.forward_actor(pi_features)
            values = self.mlp_extractor.forward_critic(vf_features)

        # Create distribution and evaluate - exactly as in SB3
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, action_log_probs, entropy
    
    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """
        Predict values for observations - matches SB3.
        """
        features = self.extract_features(obs, self.vf_features_extractor)
        if self.share_features_extractor:
            _, values = self.mlp_extractor(features)
        else:
            # When share_features_extractor is False, extract_features returns (pi_features, vf_features)
            # but since we passed self.vf_features_extractor, the first element (pi_features) is not used.
            # We only need vf_features for the critic.
            vf_features = features[1]
            values = self.mlp_extractor.forward_critic(vf_features)
        return values


def create_actor_critic(
    embedder,
    embed_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout_prob: float = 0.0,
    device: torch.device = None,
    padding_atoms: Optional[int] = None,
    padding_states: Optional[int] = None,
    max_arity: Optional[int] = None,
    total_vocab_size: Optional[int] = None,
    match_sb3_init: bool = False,
    init_seed: Optional[int] = None,
    **kwargs
) -> ActorCriticPolicy:
    """
    Factory function to create an ActorCriticPolicy.
    Matches the interface used in train.py.
    """
    policy = ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        device=device,
        share_features_extractor=kwargs.get('share_features_extractor', True),
        init_seed=init_seed,
        action_dim=int(padding_states) if padding_states is not None else None,
    )
    return policy

