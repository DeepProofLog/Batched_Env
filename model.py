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
        
        return obs_embeddings, action_embeddings, action_mask


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

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply shared residual network to embeddings.
        
        Args:
            embeddings: Input embeddings of shape [..., embed_dim].
            mask: Optional boolean mask (unused in dense path, kept for interface).
            
        Returns:
            Processed features of shape [..., hidden_dim].
        """
        original_shape = embeddings.shape
        flat_embeddings = embeddings.view(-1, original_shape[-1])  # [N, embed_dim]
        
        # Process everything (dense/padded path)
        x = self.input_transform(flat_embeddings)  # [N, hidden_dim]
        
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
        # NOTE: Inner functions (forward_policy, forward_value, forward_joint) are NOT compiled here
        # because ActorCriticPolicy.forward() is compiled with fullgraph=True, which captures
        # the entire end-to-end graph including these inner functions. Compiling both inner
        # and outer functions causes nested CUDA graph replays (443 vs ~288), hurting performance.
            
    def _encode_with_shared_body(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process embeddings through the shared backbone.
        
        Args:
            embeddings: Input embeddings [..., embed_dim].
            mask: Optional mask for embeddings.
            
        Returns:
            Processed features [..., hidden_dim].
        """
        return self.shared_body(embeddings, mask=mask)

    def forward_policy(self, obs_embeddings: torch.Tensor, 
                       action_embeddings: torch.Tensor, 
                       action_mask: torch.Tensor) -> torch.Tensor:
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
        
        shared_actions = self._encode_with_shared_body(action_embeddings, mask=action_mask)  # [B, S, H]
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
        return self.value_head(shared_obs)  # [B]

    def forward_joint(self, obs_embeddings: torch.Tensor, 
                      action_embeddings: torch.Tensor, 
                      action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both action logits and value estimate sharing the body.
        
        Args:
            obs_embeddings: Observation embeddings.
            action_embeddings: Action embeddings.
            action_mask: Action mask.
            
        Returns:
             logits, values
        """
        # Encode observations ONCE
        shared_obs = self._encode_with_shared_body(obs_embeddings)  # [B, 1, A, H]
        
        # Policy Head path
        encoded_obs = self.policy_head(shared_obs)  # [B, 1, A, E]
        shared_actions = self._encode_with_shared_body(action_embeddings, mask=action_mask)
        encoded_actions = self.policy_head(shared_actions)
        
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)
        logits = logits / (obs_embeddings.shape[-1] ** 0.5)
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        # Value Head path (reusing shared_obs)
        value = self.value_head(shared_obs)
        
        return logits, value


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
        obs_embeddings, action_embeddings, action_mask = features
        return self.shared_network.forward_joint(obs_embeddings, action_embeddings, action_mask)
    
    def forward_actor(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                      ) -> torch.Tensor:
        """Compute action logits only.
        
        Args:
            features: Tuple of (obs_embeddings, action_embeddings, action_mask).
            
        Returns:
            Action logits [B, S].
        """
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
        
        # NOTE: Policy compilation is handled at the PPO level via torch.compile(policy)
        # This creates a single unified graph instead of multiple nested graphs
        
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
        deterministic: bool = False,
        actions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass - exactly matches SB3's forward.
        
        When `actions` is None: samples actions and returns (actions, values, log_probs)
        When `actions` is provided: evaluates given actions  and returns (values, log_probs, entropy)
        
        Args:
            obs: Observations as TensorDict
            deterministic: If True, select argmax action instead of sampling
            actions: Optional actions to evaluate (for functional gradient computation)
        
        Returns:
            If actions is None: Tuple of (actions, values, log_probs)
            If actions is provided: Tuple of (values, log_probs, entropy)
        """
        # Extract features following sb3 pattern
        features = self.extract_features(obs)
        if self.share_features_extractor:
            logits, values = self.mlp_extractor(features)
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
            logits = self.mlp_extractor.forward_actor((obs_embeddings, action_embeddings, action_mask))
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))

        distribution = self.action_dist.proba_distribution(action_logits=logits)
        
        # Evaluation mode: when actions are provided, evaluate them (like evaluate_actions)
        if actions is not None:
            action_log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy()
            return values, action_log_probs, entropy
        
        # Sampling mode: sample or select deterministic action
        if deterministic:
            sampled_actions = distribution.mode()
        else:
            sampled_actions = distribution.sample()
        
        # Compute log probabilities
        action_log_probs = distribution.log_prob(sampled_actions)
        
        return sampled_actions, values, action_log_probs
    
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
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
            logits = self.mlp_extractor.forward_actor((obs_embeddings, action_embeddings, action_mask))
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))

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
            (_, _, _), (obs_embeddings_vf, _, _) = features
            values = self.mlp_extractor.forward_critic((obs_embeddings_vf, None, None))
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

