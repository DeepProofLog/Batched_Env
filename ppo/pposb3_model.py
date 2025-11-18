"""
GPU-based Policy and Value Networks for PPO using TensorDict.

This module provides actor-critic networks that mimic the SB3 models
but work entirely on GPU with torch tensors and tensordicts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from tensordict import TensorDict
from stable_baselines3.common.distributions import CategoricalDistribution
from debug_config import DebugConfig


class CustomCombinedExtractor(nn.Module):
    """
    Feature extractor that converts index-based observations into embeddings.
    This mimics the sb3 BaseFeaturesExtractor pattern.
    
    Returns a tuple of (obs_embeddings, action_embeddings, action_mask) which
    can be used by both policy and value networks.
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
        # Get tensors from TensorDict
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
    Residual MLP that produces policy logits from observation embeddings.
    
    This network processes observation embeddings and action embeddings to
    produce logits for action selection. It uses residual connections for
    better gradient flow.
    
    Args:
        embed_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
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
        
        # Residual blocks for deep feature extraction
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
    
    def _encode_embeddings(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Apply shared residual stack to embeddings. 
        
        Args:
            embeddings: Input embeddings to encode
            mask: Optional boolean mask (batch, n_items) indicating valid items.
                  If provided, only valid items are processed through the network.
        """
        batch_size, n_items, embed_dim = embeddings.shape
        # Flatten mask to 1D
        mask_flat = mask.bool().reshape(-1)
        
        # Extract only valid embeddings
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        valid_embeddings = flat_embeddings[mask_flat]
        
        # Process valid embeddings through network
        x = self.obs_transform(valid_embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        encoded_valid = self.out_transform(x)
        
        # Create output tensor and scatter valid results (match dtype for AMP)
        encoded_flat = torch.zeros_like(flat_embeddings, dtype=encoded_valid.dtype)
        encoded_flat[mask_flat] = encoded_valid
        return encoded_flat.view(batch_size, n_items, embed_dim)
    
    def forward(
        self,
        obs_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        action_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            obs_embeddings: Observation embeddings (batch, embed_dim) or (batch, 1, embed_dim)
            action_embeddings: Action embeddings (batch, n_actions, embed_dim)
            action_mask: Valid action mask (batch, n_actions)
        
        Returns:
            Masked logits (batch, n_actions)
        """
        # Process embeddings through residual network
        # Encode observations (always process all)
        encoded_obs = self._encode_embeddings(obs_embeddings)
        # Encode actions (skip invalid actions using mask for efficiency)
        encoded_actions = self._encode_embeddings(action_embeddings, mask=action_mask)
        
        # If obs has an extra dimension, squeeze it
        if encoded_obs.dim() == 3 and encoded_obs.shape[1] == 1:
            encoded_obs = encoded_obs.squeeze(1)
        
        # Compute similarity (dot product) between observation and action embeddings
        # Use bmm for efficiency when possible
        logits = torch.bmm(
            encoded_obs.unsqueeze(1),
            encoded_actions.transpose(1, 2)
        ).squeeze(1)
        
        # Scale logits by 1/sqrt(embed_dim) like in scaled dot-product attention
        # This prevents the dot products from growing too large and causing
        # the softmax to become too peaked (low entropy)
        logits = logits / (self.embed_dim ** 0.5)
        
        # Apply action mask
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits


class ValueNetwork(nn.Module):
    """
    Residual MLP that maps observation embeddings to scalar value estimates.
    
    Args:
        embed_dim: Dimension of input embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 8,
        dropout_prob: float = 0.2
    ):
        super().__init__()
        
        # Initial transformation
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
        Compute value estimates.
        
        Args:
            embeddings: Observation embeddings (batch, embed_dim) or (batch, 1, embed_dim)
        
        Returns:
            Value estimates (batch,)
        """
        # Handle shape
        if embeddings.dim() == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.squeeze(1)
        
        # Process through network
        x = self.input_layer(embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        value = self.output_layer(x)
        return value.squeeze(-1)
    
    def forward_with_encoded_obs(self, encoded_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimates from already-encoded observation embeddings.
        This skips the initial encoding step for efficiency when obs are pre-encoded.
        
        Args:
            encoded_embeddings: Pre-encoded observation embeddings (batch, embed_dim)
        
        Returns:
            Value estimates (batch,)
        """
        # Process through network (skipping initial encoding)
        x = self.input_layer(encoded_embeddings)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        
        value = self.output_layer(x)
        return value.squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """
    Combined actor-critic policy that processes observations and produces
    actions and value estimates.
    
    This class combines the policy and value networks with an embedder
    to create a complete actor-critic model.
    
    Args:
        embedder: Embedder to convert observations to embeddings
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
        device: PyTorch device
    """
    
    def __init__(
        self,
        embedder,
        embed_dim: int = 100,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_prob: float = 0.0,
        device: torch.device = None,
        debug_config: Optional[DebugConfig] = None,
        use_compile: bool = False,
        use_amp: bool = False,
        share_features_extractor: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.device = device if device is not None else torch.device('cpu')
        self.debug_config = debug_config or DebugConfig()
        self._debug_step_counter = 0
        self.use_amp = use_amp
        self.use_compile = use_compile
        self.share_features_extractor = share_features_extractor
        
        # Create features extractor(s) - following sb3 pattern
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
        
        # Create action distribution (more efficient to create once)
        # Note: action_space.n is not available here, but CategoricalDistribution
        # will determine the number of actions from the logits shape at runtime
        self.action_dist = CategoricalDistribution(action_dim=-1)
        
        # Move to device
        self.to(self.device)
        
        # Apply torch.compile if requested (PyTorch 2.0+)
        if use_compile:
            try:
                import time
                compile_start = time.time()
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                # with dynamic operations like masked indexing
                self.policy_net = torch.compile(self.policy_net, mode='default')
                self.value_net = torch.compile(self.value_net, mode='default')
                compile_time = time.time() - compile_start
                print(f"[OPTIMIZATION] torch.compile() enabled for policy and value networks (init: {compile_time:.3f}s)")
            except Exception as e:
                print(f"[WARNING] torch.compile() failed: {e}. Continuing without compilation.")
    
    def extract_features(
        self, 
        obs: TensorDict,
        features_extractor: Optional[CustomCombinedExtractor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        """
        Extract features from observations. Following sb3 pattern with share_features_extractor.
        
        Args:
            obs: TensorDict containing observations
            features_extractor: Optional extractor to use (ignored if not sharing)
        
        Returns:
            If share_features_extractor: single tuple of (obs_emb, action_emb, mask)
            Otherwise: tuple of two tuples for policy and value
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
        Forward pass: compute actions, values, and log probabilities.
        
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
        
        # Mark step boundary for CUDA graphs when using torch.compile()
        if self.use_compile:
            torch.compiler.cudagraph_mark_step_begin()
        
        # Get logits and values from compiled networks
        logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        if self.share_features_extractor:
            values = self.value_net(obs_embeddings)
        else:
            values = self.value_net(obs_embeddings_vf)
        
        # Sample actions using the pre-created distribution
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()
        
        # Compute log probabilities
        action_log_probs = distribution.log_prob(actions)
        
        # Debug output
        if self.debug_config.is_enabled('model') and self._debug_step_counter % self.debug_config.debug_sample_frequency == 0:
            self._debug_forward(logits, actions, action_log_probs, action_mask, distribution)
        self._debug_step_counter += 1
        
        return actions, values, action_log_probs
    
    def evaluate_actions(
        self,
        obs: TensorDict,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training (compute values, log probs, and entropy).
        
        Args:
            obs: Observations as TensorDict
            actions: Actions to evaluate (batch,)
        
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        # Extract features following sb3 pattern
        features = self.extract_features(obs)
        
        if self.share_features_extractor:
            obs_embeddings, action_embeddings, action_mask = features
            # OPTIMIZATION: Encode observation embeddings once and reuse for both policy and value
            # This avoids redundant computation through residual blocks
            encoded_obs = self.policy_net._encode_embeddings(obs_embeddings)
            if encoded_obs.dim() == 3 and encoded_obs.shape[1] == 1:
                encoded_obs = encoded_obs.squeeze(1)
            obs_embeddings_vf = encoded_obs
        else:
            (obs_embeddings, action_embeddings, action_mask), (obs_embeddings_vf, _, _) = features
            # Encode obs for policy
            encoded_obs = self.policy_net._encode_embeddings(obs_embeddings)
            if encoded_obs.dim() == 3 and encoded_obs.shape[1] == 1:
                encoded_obs = encoded_obs.squeeze(1)
        
        # Encode actions with masking
        encoded_actions = self.policy_net._encode_embeddings(action_embeddings, mask=action_mask)
        
        # Compute logits using encoded embeddings
        logits = torch.bmm(
            encoded_obs.unsqueeze(1),
            encoded_actions.transpose(1, 2)
        ).squeeze(1)
        logits = logits / (self.embed_dim ** 0.5)
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        # Compute values - use appropriate obs embeddings based on sharing
        if self.share_features_extractor:
            values = self.value_net.forward_with_encoded_obs(encoded_obs)
        else:
            # Encode value obs separately if not sharing
            encoded_obs_vf = self.value_net.input_layer(obs_embeddings_vf.squeeze(1) if obs_embeddings_vf.dim() == 3 else obs_embeddings_vf)
            values = self.value_net.forward_with_encoded_obs(encoded_obs_vf)
        
        # Use pre-created distribution for computing log probabilities and entropy
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        action_log_probs = distribution.log_prob(actions)
        
        # Compute entropy - the distribution handles masked actions correctly
        entropy = distribution.entropy()
        
        # Debug output for training
        if self.debug_config.is_enabled('model', level=2):
            self._debug_evaluate(logits, actions, action_log_probs, entropy, action_mask, values, distribution)
        
        # Check for NaN/Inf in results (after distribution computation)
        if torch.isnan(action_log_probs).any() or torch.isinf(action_log_probs).any():
            raise RuntimeError("NaN or Inf detected in action_log_probs in evaluate_actions()." \
                                "Check action masks and logits."
                               f"\n action_log_probs: {action_log_probs}"
                               f"\n logits: {logits}"
                               f"\n action_mask: {action_mask}"
                               f"\n actions: {actions}")
        
        return values, action_log_probs, entropy
    
    def predict_values(self, obs: TensorDict) -> torch.Tensor:
        """
        Predict values for observations.
        
        Args:
            obs: Observations as TensorDict
        
        Returns:
            Value estimates (batch,)
        """
        obs_embeddings, _, _ = self.extract_features(obs, self.vf_features_extractor)
        return self.value_net(obs_embeddings)
    
    def _debug_forward(self, logits, actions, action_log_probs, action_mask, distribution):
        """Debug output for forward pass (action selection during rollout)."""
        n_envs = min(self.debug_config.debug_sample_envs or logits.shape[0], logits.shape[0])
        
        print(f"\n{self.debug_config.debug_prefix} [MODEL FORWARD - Step {self._debug_step_counter}]")
        
        # Action mask statistics
        if self.debug_config.debug_model_action_mask:
            valid_actions = action_mask.sum(dim=-1)
            print(f"  Valid actions per env: mean={valid_actions.float().mean():.2f}, "
                  f"min={valid_actions.min()}, max={valid_actions.max()}")
        
        # Show logits for first few environments
        if self.debug_config.debug_model_logits:
            for i in range(n_envs):
                valid_mask = action_mask[i].bool()
                valid_logits = logits[i][valid_mask]
                if len(valid_logits) > 0:
                    print(f"  Env {i}: logits range=[{valid_logits.min():.3f}, {valid_logits.max():.3f}], "
                          f"mean={valid_logits.mean():.3f}, std={valid_logits.std():.3f}")
        
        # Show distribution parameters
        if self.debug_config.debug_model_distribution:
            probs = distribution.distribution.probs
            for i in range(n_envs):
                valid_mask = action_mask[i].bool()
                valid_probs = probs[i][valid_mask]
                if len(valid_probs) > 0:
                    entropy_i = distribution.entropy()[i]
                    print(f"  Env {i}: prob range=[{valid_probs.min():.4f}, {valid_probs.max():.4f}], "
                          f"entropy={entropy_i:.4f}")
        
        # Show selected actions
        if self.debug_config.debug_model_actions:
            for i in range(n_envs):
                action_i = actions[i].item()
                log_prob_i = action_log_probs[i].item()
                prob_i = torch.exp(torch.tensor(log_prob_i)).item()
                print(f"  Env {i}: selected action={action_i}, log_prob={log_prob_i:.4f}, prob={prob_i:.4f}")
    
    def _debug_evaluate(self, logits, actions, action_log_probs, entropy, action_mask, values, distribution):
        """Debug output for evaluate_actions (during training)."""
        batch_size = logits.shape[0]
        n_show = min(self.debug_config.debug_sample_envs or 5, batch_size)
        
        print(f"\n{self.debug_config.debug_prefix} [MODEL EVALUATE]")
        
        # Overall statistics
        valid_actions_per_env = action_mask.sum(dim=-1).float()
        print(f"  Batch size: {batch_size}")
        print(f"  Valid actions: mean={valid_actions_per_env.mean():.2f}, "
              f"min={valid_actions_per_env.min():.0f}, max={valid_actions_per_env.max():.0f}")
        
        # Entropy analysis (key for debugging low entropy)
        if self.debug_config.debug_model_entropy:
            print(f"  Entropy: mean={entropy.mean():.4f}, min={entropy.min():.4f}, max={entropy.max():.4f}")
            print(f"  Log prob: mean={action_log_probs.mean():.4f}, min={action_log_probs.min():.4f}, max={action_log_probs.max():.4f}")
            
            # Analyze distribution concentration
            probs = distribution.distribution.probs
            for i in range(n_show):
                valid_mask = action_mask[i].bool()
                valid_probs = probs[i][valid_mask]
                if len(valid_probs) > 0:
                    max_prob = valid_probs.max().item()
                    action_i = actions[i].item()
                    selected_prob = probs[i, action_i].item()
                    print(f"  Env {i}: entropy={entropy[i]:.4f}, max_prob={max_prob:.4f}, "
                          f"selected_action={action_i}, selected_prob={selected_prob:.4f}, "
                          f"n_valid={valid_mask.sum()}")
        
        # Values statistics
        if self.debug_config.debug_model_values:
            print(f"  Values: mean={values.mean():.4f}, min={values.min():.4f}, max={values.max():.4f}")


def create_actor_critic(
    embedder,
    embed_dim: int = 100,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout_prob: float = 0.0,
    device: torch.device = None,
    debug_config: Optional[DebugConfig] = None,
    use_compile: bool = False,
    use_amp: bool = False,
    share_features_extractor: bool = True,
) -> ActorCriticPolicy:
    """
    Factory function to create an actor-critic policy.
    
    Args:
        embedder: Embedder to convert observations to embeddings
        embed_dim: Dimension of embeddings
        hidden_dim: Dimension of hidden layers
        num_layers: Number of residual blocks
        dropout_prob: Dropout probability
        device: PyTorch device
        debug_config: Debug configuration
        use_compile: Whether to use torch.compile() for optimization (requires PyTorch 2.0+)
        use_amp: Whether to use automatic mixed precision training
        share_features_extractor: Whether to share the feature extractor between policy and value (default: True)
    
    Returns:
        ActorCriticPolicy instance
    """
    return ActorCriticPolicy(
        embedder=embedder,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        debug_config=debug_config,
        device=device,
        use_compile=use_compile,
        use_amp=use_amp,
        share_features_extractor=share_features_extractor,
    )
