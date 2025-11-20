import time
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch
import torch._dynamo
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv



# class that takes PPO but modifies the collect_rollouts method to return the actions and probs
class PPO_custom(PPO):
    """PPO variant with custom rollout collection for richer logging."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_time = time.time()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect rollouts while keeping logging consistent with the base implementation."""
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()

        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        # print('Entering rollout loop')
        while n_steps < n_rollout_steps:
            if n_steps % (n_rollout_steps // 5) == 0:
                print(f"Collecting rollouts: {n_steps}/{n_rollout_steps} steps")

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert observations to tensor and pass to policy
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            # Execute actions in environment
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    print('Rewards modified in rollout buffer because of terminal state!!!!!!!')
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs 
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Record rollout metrics for logging
        if self.ep_info_buffer and len(self.ep_info_buffer) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar=False,
    ):
        """Mirror the base learn loop but expose additional timing information."""
        total_timesteps, callback = self._setup_learn(
            total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        iteration = 0

        while self.num_timesteps < total_timesteps:
            start = time.time()
            print('Collecting rollouts')
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps
            )
            print('Time to collect_rollouts', round(time.time()-start,2))
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            
            # Train the model
            print('Training model')
            start = time.time()
            super().train()
            print('Time to train', round(time.time()-start,2))

            if not callback.on_step():
                break

            if log_interval and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self._start_time)) if self.num_timesteps > 0 else 0
                self.logger.record("time/iterations", iteration)
                self.logger.record("time/fps", fps)
                self.logger.record("time/total_timesteps", self.num_timesteps)
                self.logger.dump(self.num_timesteps)

        callback.on_training_end()
        return self


class PolicyNetwork(nn.Module):
    """Residual MLP that produces policy logits from observation embeddings."""
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=8, dropout_prob=0.2):
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
            residual = x
            x = block(x)
            x = x + residual
        encoded = self.out_transform(x)
        return encoded.view(*original_shape[:-1], -1)
    
    def forward(self, obs_embeddings, action_embeddings, action_mask):
        # Process observation embeddings through initial transformation
        encoded_obs = self._encode_embeddings(obs_embeddings)
        encoded_actions = self._encode_embeddings(action_embeddings)
        # Compute similarity (dot product) between observation and action embeddings
        logits = torch.matmul(encoded_obs, encoded_actions.transpose(-2, -1)).squeeze(-2)
        # logits = F.cosine_similarity(x, action_embeddings, dim=-1) # Compare along the embedding dimension

        # action_mask: (batch, pad_states) with 1 for valid slots
        logits = logits.masked_fill(~action_mask.bool(), float("-inf"))  

        return logits

class ValueNetwork(nn.Module):
    """Residual MLP that maps observation embeddings to scalar value estimates."""
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=8, dropout_prob=0.2):
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
    
    def forward(self, embeddings):
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



class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 1, 
        embed_dim: int = 200
    ):
        super(CustomNetwork, self).__init__()
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_network = PolicyNetwork(embed_dim)
        self.value_network = ValueNetwork(embed_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward method for SB3 compatibility.
        Accepts features (which can include sub_indices for embedding) and outputs:
        - latent policy representation
        - latent value representation
        """
        # Assuming `features` is observation sub_indices passed here for embedding
        obs_embeddings, action_embeddings, action_mask = features
        probs = self.policy_network(obs_embeddings, action_embeddings, action_mask) # (batch_size=n_envs,pad_states)
        value = self.value_network(obs_embeddings).squeeze(-1) # (batch_size=n_envs)
        return probs, value
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the actor network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent policy representation.
        """
        obs_embeddings, action_embeddings, action_mask = features
        return self.policy_network(obs_embeddings, action_embeddings, action_mask)


    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the critic network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent value representation.
        """
        obs_embeddings, _, _ = features
        return self.value_network(obs_embeddings).squeeze(-1)



class CustomCombinedExtractor(BaseFeaturesExtractor):
    """Feature extractor that converts index-based observations into embeddings."""
    def __init__(self, observation_space: gym.spaces.Dict,*args,**kwargs):
        super().__init__(observation_space, features_dim=1)

        self.embedder = kwargs.pop("embedder", None)
        if self.embedder is not None:
            self.embedding_dim = self.embedder.embed_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations including valid action mask."""

        obs_sub_indices = observations["sub_index"] # (batch_size=n_envs,1,pad_atoms,3) 2nd dim is to match the shape of derived_sub_indices 
        action_sub_indices = observations["derived_sub_indices"] # (batch_size=n_envs,pad_states,pad_atoms,3) 
        action_mask = observations["action_mask"]

        if obs_sub_indices.dtype != torch.int32:
            obs_sub_indices = obs_sub_indices.to(torch.int32)
        if action_sub_indices.dtype != torch.int32:
            action_sub_indices = action_sub_indices.to(torch.int32)

        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices) # (batch_size=n_envs,n_states=1,embedding_dim)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices) # (batch_size=n_envs,pad_states,embedding_dim)

        return obs_embeddings, action_embeddings, action_mask #, valid_actions_mask



class CustomActorCriticPolicy(MultiInputActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCombinedExtractor,
        share_features_extractor: bool = True,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        self.features_extractor_kwargs = kwargs.pop("features_extractor_kwargs", {})
        self._init_kge_settings(kwargs)
        self._init_top_k_settings(kwargs)

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class = features_extractor_class,
            share_features_extractor = share_features_extractor,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_kwargs= self.features_extractor_kwargs,
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

        self._register_special_predicates()
        self._setup_kge_bias_module()

        self.inference_second_action = False
        if self.inference_second_action:
            # Get the integer index for the 'Endf' predicate
            self.endf_pred_idx = 7 # this is just temporary, substitute by line below
            # self.endf_pred_idx = self.index_manager.predicate_str2idx.get('Endf', None)

    def _init_kge_settings(self, kwargs: Dict[str, object]) -> None:
        """Extract and document kwargs related to KGE-assisted policies."""
        self.enable_kge_action = bool(kwargs.pop("enable_kge_action", False))
        self.enable_logit_fusion = bool(kwargs.pop("enable_logit_fusion", False))
        self.kge_inference_engine = kwargs.pop("kge_inference_engine", None)
        self.index_manager = kwargs.pop("index_manager", None)
        self.debug_scores = kwargs.pop("debug_scores", False)

        # Parameters for trainable fusion MLP
        self.kge_bias_hidden_dim = kwargs.pop("kge_bias_hidden_dim", 32)

        # KGE logit gain parameters
        self.kge_logit_gain_initial = float(kwargs.pop("kge_logit_gain_initial", 0.0))
        self.kge_logit_transform = kwargs.pop("kge_logit_transform", "log")
        self.kge_logit_eps = float(kwargs.pop("kge_logit_eps", 1e-9))

        # Current gain value
        self.kge_logit_gain = self.kge_logit_gain_initial

    def _init_top_k_settings(self, kwargs: Dict[str, object]) -> None:
        """Extract and store kwargs related to top-k action pruning."""
        self.enable_top_k = bool(kwargs.pop("enable_top_k", False))
        self.top_k_value = kwargs.pop("top_k_value", None)
        self.top_k_preserve_special_predicates = kwargs.pop("top_k_preserve_special_predicates", True)
        self.debug_top_k_pruning = kwargs.pop("debug_top_k_pruning", False)
        self.top_k_debug_freq = kwargs.pop("top_k_debug_freq", 200)
        self.top_k_debug_max_envs = kwargs.pop("top_k_debug_max_envs", 3)
        self.debug_log_prob_anomalies = kwargs.pop("debug_log_prob_anomalies", True)
        if isinstance(self.top_k_value, int) and self.top_k_value <= 0:
            self.top_k_value = None

    def _extract_action_context(self, features: torch.Tensor | Tuple[torch.Tensor, ...]) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Unpack observation, action embeddings, and mask from features if available."""
        if isinstance(features, tuple):
            if len(features) == 3:
                obs_embeddings, action_embeddings, action_mask = features
                return obs_embeddings, action_embeddings, action_mask
            if len(features) == 2:
                pi_features, _ = features
                if isinstance(pi_features, tuple) and len(pi_features) == 3:
                    obs_embeddings, action_embeddings, action_mask = pi_features
                    return obs_embeddings, action_embeddings, action_mask
        return None

    def _setup_kge_bias_module(self) -> None:
        """Build the KGE bias fusion module when training it jointly with the policy.
        Used by train_bias strategy."""
        self.kge_fusion_mlp: Optional[nn.Module] = None
        if not self.enable_kge_action:
            return

        self.kge_fusion_mlp = nn.Sequential(
            nn.Linear(1, self.kge_bias_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.kge_bias_hidden_dim, 1),
        )
        self.kge_fusion_mlp.to(self.device)
        
        if hasattr(self, "optimizer") and self.optimizer is not None:
            self.optimizer.add_param_group({"params": self.kge_fusion_mlp.parameters()})


    def _register_special_predicates(self) -> None:
        """Cache indices of special predicates so they can be preserved in top-k pruning.
        Used by top_k_filtering strategy."""
        special_indices: List[int] = []
        if self.index_manager is not None:
            special_names = getattr(self.index_manager, "special_preds", [])
            predicate_lookup = getattr(self.index_manager, "predicate_str2idx", {})
            for name in special_names:
                idx = predicate_lookup.get(name)
                if idx is not None:
                    special_indices.append(int(idx))
        if special_indices:
            special_tensor = torch.tensor(special_indices, dtype=torch.int64)
        else:
            special_tensor = torch.empty(0, dtype=torch.int64)
        self.register_buffer("special_predicate_indices", special_tensor, persistent=False)


    # -------------------------------------
    # KGE TOP-K FILTERING HELPERS
    # -------------------------------------


    def _build_special_action_mask(
        self,
        obs: PyTorchObs,
        action_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Optional[torch.Tensor]:
        """Return a mask that keeps special predicates inside the Top-K selection.
        Used by top_k_filtering strategy."""
        if (
            not self.top_k_preserve_special_predicates
            or action_context is None
            or not isinstance(obs, dict)
            or self.special_predicate_indices.numel() == 0
        ):
            return None

        derived_sub_indices = obs.get("derived_sub_indices")
        _, _, action_mask = action_context

        device = action_mask.device
        specials = self.special_predicate_indices.to(device)
        predicate_indices = derived_sub_indices[:, :, 0, 0].to(device=device, dtype=specials.dtype)

        protected_mask = torch.isin(predicate_indices, specials)
        if protected_mask.shape != action_mask.shape:
            protected_mask = protected_mask.view_as(action_mask)
        protected_mask &= action_mask.to(device=device, dtype=torch.bool)

        return protected_mask if protected_mask.any() else None

    def _filter_action_logits_top_k(
        self,
        action_logits: torch.Tensor,
        action_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        forced_action_indices: Optional[torch.Tensor] = None,
        protected_action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restrict logits to the top-k actions ranked by the value network.
        Used by top_k_filtering strategy."""
        if (
            not self.enable_top_k
            or self.top_k_value is None
            or self.top_k_value <= 0
            or action_context is None
        ):
            return action_logits

        _, action_embeddings, action_mask = action_context

        mask_bool = self._normalize_action_mask(action_mask, action_embeddings)
        action_values = self.mlp_extractor.value_network(action_embeddings).detach()
        action_values = action_values.masked_fill(~mask_bool, float("-inf"))

        k = min(self.top_k_value, action_values.shape[1])
        _, topk_indices = torch.topk(action_values, k=k, dim=1)
        selection_mask = torch.zeros_like(action_values, dtype=torch.bool)
        selection_mask.scatter_(1, topk_indices, True)
        selection_mask &= mask_bool

        selection_mask = self._apply_protected_actions(selection_mask, protected_action_mask)
        selection_mask = self._apply_forced_actions(selection_mask, forced_action_indices, mask_bool)
        selection_mask = self._guard_empty_top_k_selection(selection_mask, mask_bool)

        filtered_logits = action_logits.masked_fill(~selection_mask, float("-inf"))
        filtered_logits = self._guard_fully_masked_logits(filtered_logits, action_logits)

        self._log_top_k_debug(
            mask_bool=mask_bool,
            selection_mask=selection_mask,
            topk_indices=topk_indices,
            action_values=action_values,
        )

        return filtered_logits

    @staticmethod
    def _normalize_action_mask(action_mask: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        """Ensure the action mask matches the shape expected by action embeddings.
        Used by top_k_filtering strategy."""
        mask_bool = action_mask.to(dtype=torch.bool)
        if mask_bool.ndim != action_embeddings.ndim - 1:
            mask_bool = mask_bool.reshape(action_embeddings.shape[:-1])
        return mask_bool

    @staticmethod
    def _apply_protected_actions(
        selection_mask: torch.Tensor,
        protected_action_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Guarantee that protected actions remain selectable.
        Used by top_k_filtering strategy."""
        if protected_action_mask is None:
            return selection_mask
        protection = protected_action_mask.to(device=selection_mask.device, dtype=torch.bool)
        if protection.ndim != selection_mask.ndim:
            protection = protection.reshape(selection_mask.shape)
        selection_mask |= protection
        return selection_mask

    @staticmethod
    def _apply_forced_actions(
        selection_mask: torch.Tensor,
        forced_action_indices: Optional[torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Insert actions that must remain in the support (e.g. sampled at rollout time).
        Used by top_k_filtering strategy."""
        if forced_action_indices is None:
            return selection_mask

        forced_indices = forced_action_indices.long()
        if forced_indices.ndim == 1:
            forced_indices = forced_indices.unsqueeze(1)

        sanitized = forced_indices.clone()
        valid_positions = (sanitized >= 0) & (sanitized < selection_mask.shape[1])
        if not valid_positions.any():
            return selection_mask

        sanitized = sanitized.masked_fill(~valid_positions, 0)
        forced_mask = torch.zeros_like(selection_mask, dtype=torch.bool)
        forced_mask.scatter_(1, sanitized, True)

        valid_marker = torch.zeros_like(selection_mask, dtype=torch.bool)
        valid_marker.scatter_(1, sanitized, valid_positions)
        forced_mask &= valid_marker
        forced_mask &= valid_mask

        selection_mask |= forced_mask
        return selection_mask

    @staticmethod
    def _guard_empty_top_k_selection(selection_mask: torch.Tensor, mask_bool: torch.Tensor) -> torch.Tensor:
        """Fallback to the original mask if nothing survives the top-k filter.
        Used by top_k_filtering strategy."""
        empty_selection = selection_mask.sum(dim=1) == 0
        if not empty_selection.any():
            return selection_mask
        print("[TopK] Empty selection encountered; restoring full mask for affected envs.")
        selection_mask = selection_mask.clone()
        selection_mask[empty_selection] = mask_bool[empty_selection]
        return selection_mask

    @staticmethod
    def _guard_fully_masked_logits(filtered_logits: torch.Tensor, original_logits: torch.Tensor) -> torch.Tensor:
        """Replace rows that became all -inf to avoid NaNs downstream.
        Used by top_k_filtering strategy."""
        invalid_rows = torch.isneginf(filtered_logits).all(dim=1)
        if not invalid_rows.any():
            return filtered_logits
        bad_envs = invalid_rows.nonzero(as_tuple=False).flatten().tolist()
        print("[TopK] Filtered logits contain all -inf entries for env indices:", bad_envs)
        filtered_logits = filtered_logits.clone()
        filtered_logits[invalid_rows] = original_logits[invalid_rows]
        return filtered_logits

    def _log_top_k_debug(
        self,
        *,
        mask_bool: torch.Tensor,
        selection_mask: torch.Tensor,
        topk_indices: torch.Tensor,
        action_values: torch.Tensor,
    ) -> None:
        """Optionally print debugging information about the pruning step.
        Used by top_k_filtering strategy."""
        if not self.debug_top_k_pruning:
            return
        counter = getattr(self, "_top_k_debug_counter", 0)
        if counter % max(1, self.top_k_debug_freq) != 0:
            self._top_k_debug_counter = counter + 1
            return

        with torch.no_grad():
            debug_envs = min(selection_mask.shape[0], max(1, self.top_k_debug_max_envs))
            for env_idx in range(debug_envs):
                valid_count = int(mask_bool[env_idx].sum().item())
                sel_count = int(selection_mask[env_idx].sum().item())
                top_indices = topk_indices[env_idx][:sel_count].cpu().tolist()
                top_values = action_values[env_idx, top_indices].detach().cpu().tolist() if top_indices else []
                print(f"[TopK][env {env_idx}] valid={valid_count} selected={sel_count} top_indices={top_indices} top_values={top_values}")

        self._top_k_debug_counter = counter + 1


    # -------------------------------------
    # KGE LOGIT SHAPING HELPERS
    # -------------------------------------

    def set_kge_logit_gain(self, gain: float) -> None:
        """Directly set the KGE logit gain scalar. Called by annealing callback."""
        self.kge_logit_gain = float(gain)

    def _apply_kge_logit_shaping(
        self,
        obs: PyTorchObs,
        action_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Add KGE-derived biases to logits before sampling to preserve gradients.
        Used by kge logit_shaping strategy."""
        if not self._should_apply_kge_logit_shaping(obs, action_context):
            return logits

        _, _, action_mask = action_context
        valid_mask = action_mask.to(device=logits.device, dtype=torch.bool)
        derived_sub_indices = obs.get("derived_sub_indices")
        derived = derived_sub_indices.to(logits.device)

        candidates, atom_list = self._gather_kge_candidates(derived, valid_mask)
        predictor = getattr(self.kge_inference_engine, "predict_batch", None)
        scores = predictor(atom_list)

        transformed_scores = self._transform_kge_scores(scores, logits.device, logits.dtype)
        gain_tensor = torch.as_tensor(self.kge_logit_gain, device=logits.device, dtype=logits.dtype)

        bias = self._scatter_kge_bias(atom_list, candidates, transformed_scores, logits)
        return logits + gain_tensor * bias

    def _should_apply_kge_logit_shaping(
        self,
        obs: PyTorchObs,
        action_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> bool:
        """Check if KGE logit shaping should be applied."""
        return (
            self.enable_logit_fusion
            and self.kge_inference_engine is not None
            and self.index_manager is not None
            and isinstance(obs, dict)
            and action_context is not None
            and abs(self.kge_logit_gain) > 1e-12
        )

    def _gather_kge_candidates(
        self,
        derived_sub_indices: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[Dict[str, List[Tuple[int, int]]], List[str]]:
        """Identify atoms that qualify for KGE reshaping.
        Used by kge logit_shaping strategy."""
        valid_positions = valid_mask.nonzero(as_tuple=False)
        if valid_positions.numel() == 0:
            return {}, []

        first_atoms = derived_sub_indices[:, :, 0, :]
        padding_idx = getattr(self.index_manager, "padding_idx", None)
        special_predicates = set(getattr(self.index_manager, "special_preds", []))

        candidates: Dict[str, List[Tuple[int, int]]] = {}
        for b_idx, a_idx in valid_positions.tolist():
            atom_tensor = first_atoms[b_idx, a_idx]
            
            if padding_idx is not None and int(atom_tensor[0].item()) == padding_idx:
                continue
            
            atom_str = self.index_manager.subindex_to_str(atom_tensor.detach().to("cpu"))
            if not atom_str:
                continue
            
            predicate_name = atom_str.split("(", 1)[0]
            if (predicate_name in {"True", "False"}
                or predicate_name in special_predicates
                or predicate_name.lower().startswith("end")
                or predicate_name.endswith("_kge")):
                continue
            
            candidates.setdefault(atom_str, []).append((b_idx, a_idx))

        return candidates, list(candidates.keys())

    def _transform_kge_scores(
        self,
        scores: Union[List[float], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply the selected transform to raw KGE scores.
        Used by kge logit_shaping strategy."""
        score_tensor = torch.as_tensor(scores, device=device, dtype=dtype)
        
        if (self.kge_logit_transform or "log").lower() in {"identity", "none"}:
            return torch.nan_to_num(score_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        
        clamped_scores = torch.clamp(score_tensor, min=self.kge_logit_eps)
        transformed = torch.log(clamped_scores)
        return torch.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _scatter_kge_bias(
        atom_list: List[str],
        candidates: Dict[str, List[Tuple[int, int]]],
        transformed_scores: torch.Tensor,
        reference_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Project transformed KGE scores back to the logits tensor shape.
        Used by kge logit_shaping strategy."""
        bias = torch.zeros_like(reference_logits)
        for atom_key, score in zip(atom_list, transformed_scores):
            for b_idx, a_idx in candidates[atom_key]:
                bias[b_idx, a_idx] = score
        return bias

    def _build_mlp_extractor(self) -> None:
        """
        Builds the extractor using the CustomNetwork that retains TorchRL logic.
        Used by both top_k_filtering and kge logit_shaping strategies.
        """
        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=self.features_extractor._features_dim,
            last_layer_dim_vf=1,
            embed_dim=self.features_extractor_kwargs['features_dim']
        )


    # -------------------------------------
    # KGE_ACTION BIAS 
    # -------------------------------------

    def get_kge_log_probs(self, obs: PyTorchObs, actions: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        """Replace or adjust log_prob of KGE actions using KGE scores."""
        if not self.enable_kge_action:
            return log_prob
        # Squeeze the actions tensor if needed and get the predicate indices of chosen actions
        actions_squeezed = actions.squeeze(1) if actions.ndim > 1 else actions
        batch_indices = torch.arange(actions_squeezed.shape[0], device=actions_squeezed.device)
        derived_sub_indices = obs.get("derived_sub_indices")
        chosen_action_sub_indices = derived_sub_indices[batch_indices, actions_squeezed]
        chosen_action_pred_indices = chosen_action_sub_indices[:, 0, 0]

        # Identify which chosen actions correspond to KGE-specific predicates
        # (avoid predictates like True/False/Endf and special ones)
        kge_indices_tensor = getattr(self, "kge_indices_tensor", None)
        kge_mask = torch.isin(chosen_action_pred_indices, kge_indices_tensor.to(chosen_action_pred_indices.device))
        kge_batch_indices = kge_mask.nonzero(as_tuple=False).squeeze(-1)
        
        if kge_batch_indices.numel() == 0:
            return log_prob

        atoms_to_predict, original_indices = self._prepare_kge_atoms(kge_batch_indices, chosen_action_sub_indices)
        if not atoms_to_predict:
            return log_prob

        scores = self.kge_inference_engine.predict_batch(atoms_to_predict)
        kge_log_probs = torch.log(torch.as_tensor(scores, device=log_prob.device, dtype=log_prob.dtype) + 1e-9)

        if self.debug_scores:
            for atom, score_value, log_value in zip(atoms_to_predict, scores, kge_log_probs.tolist()):
                print(f"KGE atom {atom}: score={score_value} -> logprob={log_value}")

        idx_tensor = torch.tensor(original_indices, device=log_prob.device, dtype=torch.int32)
        flat_lp = log_prob.view(-1)

        if self.kge_fusion_mlp is not None:
            kge_bias = self.kge_fusion_mlp(kge_log_probs.unsqueeze(-1)).squeeze(-1)
            flat_lp[idx_tensor] = flat_lp[idx_tensor] + kge_bias
        else:
            flat_lp[idx_tensor] = kge_log_probs
        
        return flat_lp.view_as(log_prob)

    def _prepare_kge_atoms(
        self,
        kge_batch_indices: torch.Tensor,
        chosen_action_sub_indices: torch.Tensor,
    ) -> Tuple[List[str], List[int]]:
        """Convert KGE-specific actions back to the original atoms str.
        Used by kge train_bias strategy."""
        atoms_to_predict: List[str] = []
        original_indices: List[int] = []

        predicate_lookup = getattr(self.index_manager, "predicate_idx2str", {})
        
        for batch_idx in kge_batch_indices.tolist():
            kge_action_sub_index = chosen_action_sub_indices[batch_idx, 0, :]
            kge_action_str = self.index_manager.subindex_to_str(kge_action_sub_index)
            pred_idx = int(kge_action_sub_index[0].item())
            kge_pred_str = predicate_lookup.get(pred_idx)
            
            if not (kge_action_str and kge_pred_str):
                continue
            
            # Remove '_kge' suffix to get original predicate
            original_pred_str = kge_pred_str.removesuffix("_kge")
            original_atom_str = f"{original_pred_str}{kge_action_str[len(kge_pred_str):]}"
            atoms_to_predict.append(original_atom_str)
            original_indices.append(int(batch_idx))

        return atoms_to_predict, original_indices

    @torch._dynamo.disable
    def _get_second_best_log_prob(self, obs: PyTorchObs, actions: torch.Tensor, logits: torch.Tensor, distribution: Distribution, original_log_prob: torch.Tensor) -> torch.Tensor:
        """
        If 'Endf' is chosen, replaces its log_prob with the log_prob of the second-best action.
        This function will now run in eager mode, avoiding compilation errors.
        """
        # Find which of the *chosen* actions correspond to 'Endf'
        action_pred_indices = obs["derived_sub_indices"][:, :, 0, 0]
        is_endf_action_matrix = (action_pred_indices == self.endf_pred_idx)
        chosen_action_is_endf = torch.gather(is_endf_action_matrix, 1, actions.unsqueeze(-1)).squeeze(-1)

        # Proceed if any 'Endf' action was actually chosen
        if torch.any(chosen_action_is_endf):
            # Get the batch indices where 'Endf' was the chosen action
            endf_chosen_indices = torch.where(chosen_action_is_endf)[0]

            # --- REVISED LOGIC START ---

            # 1. Select the logits for only the environments that chose 'Endf'
            selected_logits = logits[endf_chosen_indices]

            # 2. To find the second-best action, temporarily set the logit of the 'Endf' action to -inf
            # We need to get the chosen actions for this specific subset of environments
            selected_actions = actions[endf_chosen_indices]
            
            logits_clone = selected_logits.clone()
            # The indices for this in-place modification must be 0-indexed within the sub-batch
            batch_idx_for_clone = torch.arange(len(endf_chosen_indices), device=logits.device)
            logits_clone[batch_idx_for_clone, selected_actions] = -float('inf')

            # 3. Get the indices of the new best actions (which are the original second-best)
            _, second_best_action_indices = logits_clone.max(dim=1)

            # 4. Manually calculate log_softmax on the selected logits and gather the values.
            # This avoids the broadcasting issue with the original distribution object.
            selected_log_probs_all_actions = F.log_softmax(selected_logits, dim=1)
            second_best_log_probs = torch.gather(selected_log_probs_all_actions, 1, second_best_action_indices.unsqueeze(-1)).squeeze(-1)

            # --- REVISED LOGIC END ---
            print("\n--- [Verbose] Score Substitution Triggered ---")
            # Loop through each instance in the batch where the substitution happened
            for i, batch_idx in enumerate(endf_chosen_indices):
                original_lp = original_log_prob[batch_idx].item()
                new_lp = second_best_log_probs[i].item()
                print(f"  Instance in Batch (Index {batch_idx.item()}): 'Endf' action was chosen.")
                print(f"    - Original 'Endf' LogProb: {original_lp:.4f}")
                print(f"    - Second-Best LogProb:     {new_lp:.4f} (Substituting)")
            print("--------------------------------------------")
            # 5. Now, this assignment will work because both sides have the same shape.
            modified_log_prob = original_log_prob.clone()
            modified_log_prob[endf_chosen_indices] = second_best_log_probs
            return modified_log_prob

        return original_log_prob

    def _debug_log_prob_anomalies(
        self,
        log_prob: torch.Tensor,
        action_logits: torch.Tensor,
        prefix: str,
        message: str,
        logits_heading: str,
        actions: Optional[torch.Tensor] = None,
        show_shapes: bool = False,
        show_actions: bool = False,
    ) -> None:
        """Mirror the verbose anomaly logging without duplicating code.
        Used by both forward and evaluate_actions methods."""
        if not self.debug_log_prob_anomalies:
            return
        anomaly_mask = torch.isnan(log_prob) | torch.isinf(log_prob)
        if not anomaly_mask.any():
            return

        bad_idx = anomaly_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
        if show_shapes and actions is not None:
            print(f"{prefix} LogProb shape:", log_prob.shape, "Actions shape:", actions.shape)
        print(f"{prefix} {message} batch indices:", bad_idx)
        print(f"{prefix} {logits_heading}")
        for idx in bad_idx[:5]:
            row = action_logits[idx]
            contains_inf = torch.isinf(row).any().item()
            contains_nan = torch.isnan(row).any().item()
            print(
                f"  idx {idx} logits min={float(row.min())} max={float(row.max())} "
                f"contains_inf={contains_inf} contains_nan={contains_nan}"
            )
        if show_actions and actions is not None and bad_idx:
            idx_tensor = torch.tensor(bad_idx, device=actions.device)
            print(f"{prefix} Actions for bad batch entries:", actions[idx_tensor])

    def forward(self, obs: torch.Tensor, deterministic: bool = False, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass in actor and critic networks."""
        del verbose
        features = self.extract_features(obs)
        action_context = self._extract_action_context(features)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features) # (batch_size=n_envs,n_states=1,embedding_dim)
            latent_vf = self.mlp_extractor.forward_critic(vf_features) # (batch_size=n_envs,n_states=1)

        special_action_mask = self._build_special_action_mask(obs, action_context)
        # Get value for the state
        values = latent_vf # (batch_size=n_envs,n_states=1)
        # Get action probabilities and create distribution
        shaped_logits = self._apply_kge_logit_shaping(obs, action_context, latent_pi)
        action_logits = self._filter_action_logits_top_k(
            shaped_logits,
            action_context,
            protected_action_mask=special_action_mask,
        )

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Sample a single action from the distribution
        actions = distribution.get_actions(deterministic=deterministic)
        # Get log probability of the chosen action
        log_prob = distribution.log_prob(actions)

        self._debug_log_prob_anomalies(
            log_prob=log_prob,
            action_logits=action_logits,
            prefix="[LogProb]",
            message="Detected nan/inf during forward.",
            logits_heading="Corresponding action logits row snapshot:",
            actions=actions,
            show_shapes=True,
            show_actions=True,
        )

        if self.enable_kge_action:
            log_prob = self.get_kge_log_probs(obs, actions, log_prob)
        
        if self.inference_second_action:
            log_prob = self._get_second_best_log_prob(obs, actions, action_logits, distribution, log_prob)
        
        return actions, values, log_prob


    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions according to current policy."""
        features = self.extract_features(obs)
        action_context = self._extract_action_context(features)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Create action distribution using logits
        raw_action_logits = self._apply_kge_logit_shaping(obs, action_context, latent_pi)
        actions_long = actions.long()
        special_action_mask = self._build_special_action_mask(obs, action_context)
        action_logits = self._filter_action_logits_top_k(
            raw_action_logits,
            action_context,
            forced_action_indices=actions_long,
            protected_action_mask=special_action_mask,
        )

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Get log probability of the given actions
        log_prob = distribution.log_prob(actions)
        values = latent_vf
        entropy = distribution.entropy()

        self._debug_log_prob_anomalies(
            log_prob=log_prob,
            action_logits=action_logits,
            prefix="[LogProb][Eval]",
            message="Detected nan/inf when evaluating actions.",
            logits_heading="Action logits summary for affected rows:",
        )

        if self.enable_kge_action:
            log_prob = self.get_kge_log_probs(obs, actions, log_prob)

        if self.inference_second_action:
            log_prob = self._get_second_best_log_prob(obs, actions, action_logits, distribution, log_prob)

        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """Get estimated values according to current policy."""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return latent_vf

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """Get current policy distribution given observations."""
        features = super().extract_features(obs, self.pi_features_extractor)
        action_context = self._extract_action_context(features)
        latent_pi = self.mlp_extractor.forward_actor(features)
        special_action_mask = self._build_special_action_mask(obs, action_context)
        shaped_logits = self._apply_kge_logit_shaping(obs, action_context, latent_pi)
        action_logits = self._filter_action_logits_top_k(
            shaped_logits,
            action_context,
            protected_action_mask=special_action_mask,
        )
        return self.action_dist.proba_distribution(action_logits=action_logits)