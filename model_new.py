"""
model_new.py - Clean implementation that exactly mimics SB3's model logic.

This file is a simplified version of model.py that removes all batched-specific
optimizations and matches the SB3 implementation as closely as possible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import gymnasium as gym
import numpy as np
from tensordict import TensorDict
from stable_baselines3.common.distributions import CategoricalDistribution
from sb3.sb3_model import (
    CustomActorCriticPolicy as SB3ActorCriticPolicy,
    CustomCombinedExtractor as SB3CombinedExtractor,
)


class CustomCombinedExtractor(nn.Module):
    """
    Feature extractor that converts index-based observations into embeddings.
    Matches SB3's BaseFeaturesExtractor pattern exactly.
    """
    
    def __init__(self, embedder):
        super().__init__()
        self.embedder = embedder
        self.embed_dim = embedder.embedding_dim
    
    def forward(self, obs: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from observations.
        
        Args:
            obs: TensorDict containing 'sub_index', 'derived_sub_indices', 'action_mask'
        
        Returns:
            Tuple of (obs_embeddings, action_embeddings, action_mask)
        """
        # Get tensors from TensorDict - exactly as in SB3
        obs_sub_indices = obs.get("sub_index")  # (batch, 1, pad_atoms, 3)
        action_sub_indices = obs.get("derived_sub_indices")  # (batch, pad_states, pad_atoms, 3)
        action_mask = obs.get("action_mask")  # (batch, pad_states)
        
        # Ensure correct dtype
        if obs_sub_indices.dtype != torch.int32:
            obs_sub_indices = obs_sub_indices.to(torch.int32)
        if action_sub_indices.dtype != torch.int32:
            action_sub_indices = action_sub_indices.to(torch.int32)
        
        # Get embeddings from embedder
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)
        
        return obs_embeddings, action_embeddings, action_mask


class PolicyNetwork(nn.Module):
    """
    Policy network that produces logits from observation and action embeddings.
    Matches SB3's implementation exactly - NO SCALING, simple processing.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Initial transformation from embedding to hidden representation
        self.obs_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks - match SB3 exactly
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final transformation back to embedding space
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def _forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP."""
        x = self.obs_transform(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        return self.out_transform(x)

    def _encode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Process embeddings through residual network.
        Matches SB3 exactly - process all embeddings, no masking optimization.
        """
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, original_shape[-1])
        encoded = self._forward_mlp(flat_embeddings)
        return encoded.view(*original_shape[:-1], -1)
    
    def forward(
        self,
        obs_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits - exactly as in SB3.
        
        Args:
            obs_embeddings: Observation embeddings (batch, embed_dim) or (batch, 1, embed_dim)
            action_embeddings: Action embeddings (batch, n_actions, embed_dim)
            action_mask: Valid action mask (batch, n_actions)
        
        Returns:
            Masked logits (batch, n_actions)
        """
        # Process embeddings through residual network
        encoded_obs = self._encode_embeddings(obs_embeddings)
        encoded_actions = self._encode_embeddings(action_embeddings)
        
        # Compute similarity (dot product) between observation and action embeddings
        # Exactly as in SB3: matmul + squeeze, NO SCALING
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)
        
        # Apply action mask
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits


class ValueNetwork(nn.Module):
    """
    Value network that maps observation embeddings to scalar value estimates.
    Matches SB3 implementation exactly.
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        
        # Initial transformation - match SB3 exactly
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from observation embeddings.
        Matches SB3 exactly.
        """
        # Process observation embeddings through the input layer
        x = self.input_layer(embeddings)
        # Pass through residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        # Get final value prediction
        value = self.output_layer(x)
        return value.squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """
    Combined actor-critic policy that exactly matches SB3's ActorCriticPolicy.
    
    This is a clean implementation that removes all batched-specific optimizations
    and matches the SB3 logic line-by-line.
    """
    
    def __init__(
        self,
        embedder,
        embed_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_prob: float = 0.0,
        device: torch.device = None,
        share_features_extractor: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.device = device if device is not None else torch.device('cpu')
        self.share_features_extractor = share_features_extractor
        
        # Create features extractor(s) - following sb3 pattern exactly
        self.features_extractor = CustomCombinedExtractor(embedder)
        if share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            # Create separate extractors for policy and value
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = CustomCombinedExtractor(embedder)
        
        # Create policy and value networks
        self.policy_net = PolicyNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        self.value_net = ValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        # Create action distribution - matches SB3
        self.action_dist = CategoricalDistribution(action_dim=-1)
        
        # Move to device
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
            obs_embeddings, action_embeddings, action_mask = features
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
        
        # Get logits and values - exactly as in SB3
        logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        if self.share_features_extractor:
            values = self.value_net(obs_embeddings)
        else:
            values = self.value_net(obs_embeddings_vf)
        
        # Sample actions using the distribution - matches SB3
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
            obs_embeddings, action_embeddings, action_mask = features
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
        
        # Get logits and values
        logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        if self.share_features_extractor:
            values = self.value_net(obs_embeddings)
        else:
            values = self.value_net(obs_embeddings_vf)
        
        # Create distribution and evaluate - exactly as in SB3
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        action_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return values, action_log_probs, entropy
    
    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """
        Predict values for observations - matches SB3.
        """
        obs_embeddings, _, _ = self.extract_features(obs, self.vf_features_extractor)
        return self.value_net(obs_embeddings)


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
    match_sb3_init: bool = True,
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
    )
    if match_sb3_init and all(v is not None for v in (padding_atoms, padding_states, max_arity, total_vocab_size)):
        _copy_weights_from_sb3(
            policy,
            embedder,
            padding_atoms=int(padding_atoms),
            padding_states=int(padding_states),
            max_arity=int(max_arity),
            total_vocab_size=int(total_vocab_size),
            init_seed=init_seed,
        )
    return policy


def _copy_weights_from_sb3(
    policy: ActorCriticPolicy,
    embedder,
    padding_atoms: int,
    padding_states: int,
    max_arity: int,
    total_vocab_size: int,
    init_seed: Optional[int] = None,
) -> None:
    """Initialize policy weights to match SB3's CustomActorCriticPolicy."""

    if not hasattr(embedder, "embed_dim") and hasattr(embedder, "embedding_dim"):
        embedder.embed_dim = embedder.embedding_dim
    features_dim = getattr(embedder, "embed_dim", None)
    if features_dim is None:
        return
    sb3_state = _build_sb3_state_dict(
        embedder=embedder,
        features_dim=features_dim,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        max_arity=max_arity,
        total_vocab_size=total_vocab_size,
        init_seed=init_seed,
    )
    mapped_state = {}
    for key in policy.state_dict().keys():
        if key.startswith("policy_net"):
            mapped_key = "mlp_extractor.policy_network" + key[len("policy_net") :]
        elif key.startswith("value_net"):
            mapped_key = "mlp_extractor.value_network" + key[len("value_net") :]
        else:
            mapped_key = key
        mapped_state[key] = sb3_state[mapped_key]

    policy.load_state_dict(mapped_state)


def _build_sb3_state_dict(
    embedder,
    features_dim: int,
    padding_atoms: int,
    padding_states: int,
    max_arity: int,
    total_vocab_size: int,
    init_seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Create a fresh SB3 CustomActorCriticPolicy and return its state dict
    deterministically (optionally seeding a forked RNG so global RNG is
    untouched). This keeps batched and SB3 initializations in sync.
    """
    def _make_policy():
        if init_seed is not None:
            torch.manual_seed(init_seed)

        obs_space = gym.spaces.Dict(
            {
                "sub_index": gym.spaces.Box(
                    low=0,
                    high=total_vocab_size,
                    shape=(1, padding_atoms, max_arity + 1),
                    dtype=np.int32,
                ),
                "derived_sub_indices": gym.spaces.Box(
                    low=0,
                    high=total_vocab_size,
                    shape=(padding_states, padding_atoms, max_arity + 1),
                    dtype=np.int32,
                ),
                "action_mask": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(padding_states,),
                    dtype=np.uint8,
                ),
            }
        )
        action_space = gym.spaces.Discrete(padding_states)

        def _schedule(_: float) -> float:
            return 0.0

        sb3_policy = SB3ActorCriticPolicy(
            obs_space,
            action_space,
            _schedule,
            features_extractor_class=SB3CombinedExtractor,
            features_extractor_kwargs={"features_dim": features_dim, "embedder": embedder},
            enable_top_k=False,
            enable_kge_action=False,
        )
        return sb3_policy.state_dict()

    if init_seed is not None:
        with torch.random.fork_rng(devices=[]):
            return _make_policy()
    return _make_policy()


def get_sb3_policy_state_dict_aligned(
    embedder,
    padding_atoms: int,
    padding_states: int,
    max_arity: int,
    total_vocab_size: int,
    init_seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Public helper for other modules (e.g., sb3_train) to obtain the deterministic
    SB3 policy state dict used for batched initialization.
    """
    if not hasattr(embedder, "embed_dim") and hasattr(embedder, "embedding_dim"):
        embedder.embed_dim = embedder.embedding_dim
    features_dim = getattr(embedder, "embed_dim", None)
    if features_dim is None:
        raise ValueError("Embedder must expose embed_dim for aligned init.")
    return _build_sb3_state_dict(
        embedder=embedder,
        features_dim=features_dim,
        padding_atoms=padding_atoms,
        padding_states=padding_states,
        max_arity=max_arity,
        total_vocab_size=total_vocab_size,
        init_seed=init_seed,
    )
