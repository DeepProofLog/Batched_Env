"""
Neural Networks for PPO

This module contains the policy and value network architectures,
as well as TorchRL-compatible wrappers for the actor-critic model.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import MaskedCategorical
from tensordict.nn import TensorDictSequential


class PolicyNetwork(nn.Module):
    """Residual MLP that produces policy logits from observation embeddings."""
    
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128, num_layers: int = 8, dropout_prob: float = 0.2):
        super().__init__()
        # Initial transformation from observation embedding to hidden representation
        self.obs_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final transformation that projects the processed observation back to the embedding space
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        
        # Residual blocks for deep feature extraction
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        
        # Final output layers to produce a single scalar value estimate
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
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
        
        # Optimize device transfer: only transfer if needed
        # Check if already on correct device to avoid unnecessary transfers
        if obs_sub_indices.device != self.device or obs_sub_indices.dtype != torch.long:
            obs_sub_indices = obs_sub_indices.to(device=self.device, dtype=torch.long, non_blocking=True)
        if action_sub_indices.device != self.device or action_sub_indices.dtype != torch.long:
            action_sub_indices = action_sub_indices.to(device=self.device, dtype=torch.long, non_blocking=True)
        if action_mask.device != self.device:
            action_mask = action_mask.to(device=self.device, non_blocking=True)
        
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


class TorchRLActorModule(nn.Module):
    """
    TorchRL-compatible actor module that wraps the actor-critic model.
    
    This module is designed to be used with TorchRL's TensorDictModule.
    """
    
    def __init__(self, actor_critic_model: ActorCriticModel):
        """
        Initialize the actor module.
        
        Args:
            actor_critic_model: The underlying actor-critic model
        """
        super().__init__()
        self.actor_critic_model = actor_critic_model
    
    def forward(
        self,
        sub_index: torch.Tensor,
        derived_sub_indices: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that produces action logits.
        
        Args:
            sub_index: Observation indices
            derived_sub_indices: Action indices
            action_mask: Valid action mask
            
        Returns:
            Tuple (logits, mask) for distribution construction
        """
        tensordict = TensorDict({
            "sub_index": sub_index,
            "derived_sub_indices": derived_sub_indices,
            "action_mask": action_mask,
        }, batch_size=[])
        logits = self.actor_critic_model.forward_actor(tensordict)
        return logits, action_mask


class TorchRLValueModule(nn.Module):
    """
    TorchRL-compatible value module that wraps the actor-critic model.
    
    This module is designed to be used with TorchRL's TensorDictModule.
    """
    
    def __init__(self, actor_critic_model: ActorCriticModel):
        """
        Initialize the value module.
        
        Args:
            actor_critic_model: The underlying actor-critic model
        """
        super().__init__()
        self.actor_critic_model = actor_critic_model

    def forward(self, sub_index: torch.Tensor) -> torch.Tensor:
        td = TensorDict({"sub_index": sub_index}, batch_size=[])
        return self.actor_critic_model.forward_critic(td)


class AddFeatureDimension(TensorDictModule):
    """
    Module that adds a feature dimension to scalar tensors.
    This is needed for TorchRL's PPO loss which expects all tensors to have a feature dimension.
    """
    def __init__(self):
        # Create a simple passthrough module
        super().__init__(
            module=nn.Identity(),
            in_keys=["sample_log_prob"],
            out_keys=["sample_log_prob"],
        )
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Add feature dimension if needed."""
        if "sample_log_prob" in tensordict.keys():
            log_prob = tensordict.get("sample_log_prob")
            if log_prob.dim() == 1:
                # Add feature dimension: [batch] -> [batch, 1]
                tensordict.set("sample_log_prob", log_prob.unsqueeze(-1))
        return tensordict


def create_torchrl_modules(
    embedder: Any,
    num_actions: int,
    embed_dim: int = 200,
    hidden_dim: int = 128,
    num_layers: int = 8,
    dropout_prob: float = 0.2,
    device: torch.device = torch.device('cpu'),
    enable_kge_action: bool = False,
    kge_inference_engine: Optional[Any] = None,
    index_manager: Optional[Any] = None,
) -> Tuple[ProbabilisticActor, ValueOperator]:
    """
    Create TorchRL-compatible actor and value modules.
    
    Args:
        embedder: Embedding model
        num_actions: Number of possible actions
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension for networks
        num_layers: Number of residual layers
        dropout_prob: Dropout probability
        device: Device to use
        enable_kge_action: Whether to enable KGE actions
        kge_inference_engine: KGE engine
        index_manager: Index manager
        
    Returns:
        Tuple of (actor_module, value_module)
    """
    # Create the shared actor-critic model
    actor_critic_model = ActorCriticModel(
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
    
    # Create actor module that outputs logits
    actor_net = TorchRLActorModule(actor_critic_model)
    
    # Wrap actor in TensorDictModule
    actor_td_module = TensorDictModule(
        module=actor_net,
        in_keys=["sub_index", "derived_sub_indices", "action_mask"],
        out_keys=["logits", "mask"],
    )
    
    # Wrap in ProbabilisticActor with MaskedCategorical so entropy/log-probs
    # remain finite even when most actions are invalid.
    actor_module = ProbabilisticActor(
        module=actor_td_module,
        in_keys={"logits": "logits", "mask": "mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Create value module
    value_net = TorchRLValueModule(actor_critic_model)

    value_module = TensorDictModule(
        module=value_net,
        in_keys=["sub_index"],            # was ["sub_index", "derived_sub_indices", "action_mask"]
        out_keys=["state_value"],
    )
    
    return actor_module, value_module
