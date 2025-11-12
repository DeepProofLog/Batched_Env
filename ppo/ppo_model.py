from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import nn



class PolicyNetwork(nn.Module):
    """Residual MLP that produces policy logits from observation embeddings."""
    
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 8, dropout_prob: float = 0.2):
        super().__init__()
        # Initial transformation from observation embedding to hidden representation
        self.obs_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),  # Inplace for memory efficiency
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction - OPTIMIZED: removed LayerNorm
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final transformation that projects the processed observation back to the embedding space
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

    def _encode_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply shared residual stack to observation or action embeddings."""
        original_shape = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, original_shape[-1])
        x = self.obs_transform(flat_embeddings)
        for block in self.res_blocks:
            x = block(x) + x  # Fused residual addition
        encoded = self.out_transform(x)
        return encoded.view(*original_shape[:-1], -1)
    
    def forward(self, obs_embeddings: torch.Tensor, action_embeddings: torch.Tensor, 
                action_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute policy logits from observation and action embeddings.
        
        Args:
            obs_embeddings: Tensor of shape (batch, embed_dim)
            action_embeddings: Tensor of shape (batch, num_actions, embed_dim)
            action_mask: Boolean tensor of shape (batch, num_actions)
            
        Returns:
            Logits tensor of shape (batch, num_actions)
        """
        # Process observation embeddings through initial transformation
        encoded_obs = self._encode_embeddings(obs_embeddings)
        encoded_actions = self._encode_embeddings(action_embeddings)

        # Compute similarity (dot product) between observation and action embeddings
        # Use bmm instead of matmul for better performance
        logits = torch.bmm(encoded_obs.unsqueeze(1), encoded_actions.transpose(1, 2)).squeeze(1)      
        # Mask invalid actions with -inf (avoid .to() call by ensuring mask is on correct device)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        
        return logits


class ValueNetwork(nn.Module):
    """Residual MLP that maps observation embeddings to scalar value estimates."""
    
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 8, dropout_prob: float = 0.2):
        super().__init__()
        # Initial transformation from observation embedding to hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),  # Inplace for memory efficiency
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction - OPTIMIZED: removed LayerNorm
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final output layers to produce a single scalar value estimate
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from observation embeddings.
        
        Args:
            embeddings: Tensor of shape (batch, embed_dim)
            
        Returns:
            Value tensor of shape (batch,)
        """
        # Process observation embeddings through the input layer
        x = self.input_layer(embeddings)
        # Pass through residual blocks with fused addition
        for block in self.res_blocks:
            x = block(x) + x  # Fused residual addition
        # Get final value prediction
        value = self.output_layer(x)
        return value.squeeze(-1)


class EmbeddingExtractor(nn.Module):
    """
    Feature extractor that converts index-based observations into embeddings.
    
    This module extracts embeddings for observations and actions using an embedder.
    """
    
    def __init__(self, embedder: Any, device: torch.device = torch.device('cpu')):
        """
        Initialize the embedding extractor.
        
        Args:
            embedder: Embedding model with get_embeddings_batch method
            device: Device to use for tensors
        """
        super().__init__()
        self.embedder = embedder
        # Try to get embedding dimension, use default if not available
        self.embedding_dim = getattr(embedder, 'embed_dim', getattr(embedder, 'atom_embedding_size', 64))
        self.device = device
    
    def forward(self, observations: TensorDict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings from observations.
        
        Args:
            observations: TensorDict containing:
                - sub_index: (batch, 1, padding_atoms, 3) observation indices
                - derived_sub_indices: (batch, padding_states, padding_atoms, 3) action indices
                - action_mask: (batch, padding_states) valid action mask
                
        Returns:
            Tuple of (obs_embeddings, action_embeddings, action_mask)
        """
        obs_sub_indices = observations["sub_index"]
        action_sub_indices = observations["derived_sub_indices"]
        action_mask = observations["action_mask"]
        
        # OPTIMIZATION: Check device once and batch transfer if needed
        needs_transfer = (obs_sub_indices.device != self.device)
        
        if needs_transfer:
            obs_sub_indices = obs_sub_indices.to(device=self.device, dtype=torch.long, non_blocking=True)
            action_sub_indices = action_sub_indices.to(device=self.device, dtype=torch.long, non_blocking=True)
            action_mask = action_mask.to(device=self.device, non_blocking=True)
        else:
            # Ensure correct dtype without transfer overhead
            if obs_sub_indices.dtype != torch.long:
                obs_sub_indices = obs_sub_indices.long()
            if action_sub_indices.dtype != torch.long:
                action_sub_indices = action_sub_indices.long()
        
        # Get embeddings
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)  # (batch, 1, embedding_dim)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)  # (batch, padding_states, embedding_dim)
        
        # Squeeze observation embeddings to (batch, embedding_dim)
        obs_embeddings = obs_embeddings.squeeze(1)
        
        return obs_embeddings, action_embeddings, action_mask

    def forward_obs(self, obs_sub_indices: torch.Tensor) -> torch.Tensor:
        if obs_sub_indices.device != self.device or obs_sub_indices.dtype != torch.long:
            obs_sub_indices = obs_sub_indices.to(device=self.device, dtype=torch.long, non_blocking=True)
        return self.embedder.get_embeddings_batch(obs_sub_indices).squeeze(1)

class ActorCriticModel(nn.Module):
    """
    Combined actor-critic model for TorchRL.
    
    This model contains both the policy (actor) and value (critic) networks,
    along with the embedding extractor.
    """
    
    def __init__(
        self,
        embedder: Any,
        embed_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2,
        device: torch.device = torch.device('cpu'),
        enable_kge_action: bool = False,
        kge_inference_engine: Optional[Any] = None,
        index_manager: Optional[Any] = None,
    ):
        """
        Initialize the actor-critic model.
        
        Args:
            embedder: Embedding model with get_embeddings_batch method
            embed_dim: Dimension of embeddings
            hidden_dim: Hidden dimension for networks
            num_layers: Number of residual layers
            dropout_prob: Dropout probability
            device: Device to use
            enable_kge_action: Whether to enable KGE-based action selection
            kge_inference_engine: KGE engine for scoring actions
            index_manager: Index manager for predicate/constant lookups
        """
        super().__init__()
        
        self.device = device
        self.embed_dim = embed_dim
        self.enable_kge_action = enable_kge_action
        self.kge_inference_engine = kge_inference_engine
        self.index_manager = index_manager
        
        # Feature extractor
        self.feature_extractor = EmbeddingExtractor(embedder, device=device)
        
        # Policy and value networks
        self.policy_network = PolicyNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        self.value_network = ValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_prob=dropout_prob
        )
        
        # Move to device
        self.to(device)
    
    def forward_actor(self, observations: TensorDict) -> torch.Tensor:
        """
        Forward pass for the actor (policy) network.
        
        Args:
            observations: TensorDict containing observation data
            
        Returns:
            Logits tensor for action distribution
        """
        obs_embeddings, action_embeddings, action_mask = self.feature_extractor(observations)
        logits = self.policy_network(obs_embeddings, action_embeddings, action_mask)
        return logits
    

    def forward_critic(self, observations: TensorDict) -> torch.Tensor:
        obs_embeddings = self.feature_extractor.forward_obs(observations["sub_index"])
        return self.value_network(obs_embeddings)

    def forward(self, observations: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both actor and critic networks.
        
        Args:
            observations: TensorDict containing observation data
            
        Returns:
            Tuple of (logits, values)
        """
        obs_embeddings, action_embeddings, action_mask = self.feature_extractor(observations)
        logits = self.policy_network(obs_embeddings, action_embeddings, action_mask)
        values = self.value_network(obs_embeddings)
        return logits, values


class OptimizedActorModule(nn.Module):
    """Lightweight actor wrapper that outputs logits and optionally samples actions."""

    def __init__(self, actor_critic_model: ActorCriticModel, deterministic: bool = True):
        """
        Initialize the actor module.
        
        Args:
            actor_critic_model: The underlying actor-critic model
            deterministic: If True, use argmax for action selection in eval mode.
                          If False, sample from the distribution in eval mode.
        """
        super().__init__()
        self.actor_critic_model = actor_critic_model
        self.deterministic = deterministic

    def forward(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.select("sub_index", "derived_sub_indices", "action_mask")
        logits = self.actor_critic_model.forward_actor(obs.clone().to(self.actor_critic_model.device))
        logits = logits.to(obs.device if obs.device is not None else logits.device)
        out = tensordict.clone(False)
        out.set("logits", logits)
        
        # For evaluation, also select actions
        if not self.training:
            action_mask = tensordict.get("action_mask")
            if action_mask is not None:
                # Ensure mask is boolean and on correct device
                mask = action_mask.to(logits.device, non_blocking=True).bool()
                # Mask invalid actions with -inf
                masked_logits = logits.masked_fill(~mask, float("-inf"))
                
                if self.deterministic:
                    # Deterministic: take argmax (most likely action)
                    action = masked_logits.argmax(dim=-1)
                else:
                    # Stochastic: sample from distribution
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = dist.sample()
                
                out.set("action", action)
                
                # Compute log_prob for the selected action
                dist = torch.distributions.Categorical(logits=masked_logits)
                log_prob = dist.log_prob(action)
                out.set("sample_log_prob", log_prob)
        
        return out


class OptimizedValueModule(nn.Module):
    """Value wrapper that produces 'state_value' entries compatible with the collector."""

    def __init__(self, actor_critic_model: ActorCriticModel):
        super().__init__()
        self.actor_critic_model = actor_critic_model

    def forward(self, tensordict: TensorDict) -> TensorDict:
        obs = tensordict.select("sub_index")
        values = self.actor_critic_model.forward_critic(obs.clone().to(self.actor_critic_model.device))
        values = values.squeeze(-1).to(obs.device if obs.device is not None else values.device)
        out = tensordict.clone(False)
        out.set("state_value", values)
        return out


def create_torch_modules(
    *,
    embedder: Any,
    num_actions: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout_prob: float,
    device: torch.device,
    enable_kge_action: bool = False,
    kge_inference_engine: Optional[Any] = None,
    index_manager: Optional[Any] = None,
) -> Tuple[nn.Module, nn.Module, ActorCriticModel]:
    """Factory for the optimised actor/critic pair."""
    core_model = ActorCriticModel(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        device=device,
        enable_kge_action=enable_kge_action,
        kge_inference_engine=kge_inference_engine,
        index_manager=index_manager,
    )

    actor_module = OptimizedActorModule(core_model)
    value_module = OptimizedValueModule(core_model)
    return actor_module, value_module