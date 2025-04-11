# --- Imports (Assume necessary imports like torch, nn, gym, stable_baselines3 types are present) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from typing import List, Union, Dict, Type, Optional, Callable, Tuple, Any
import time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy, ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import (Distribution, make_proba_distribution, CategoricalDistribution) # Ensure CategoricalDistribution is imported
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer # Use DictRolloutBuffer for Dict observations
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule



# # --- PPO_custom needs adjustment for action shape ---
# class PPO_custom(PPO):
#     """
#     Custom PPO class that modifies collect_rollouts for the batch environment.
#     Handles reshaping of actions and observations.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._start_time = time.time()
#         # Ensure DictRolloutBuffer is used
#         if isinstance(self.observation_space, spaces.Dict):
#             buffer_size = self.n_steps * self.n_envs # n_envs is 1, n_steps is e.g. 128
#             self.rollout_buffer = DictRolloutBuffer(
#                 buffer_size, # This size seems small? SB3 might adjust internally.
#                 self.observation_space,
#                 self.action_space, # Original action space (MultiDiscrete?)
#                 device=self.device,
#                 gamma=self.gamma,
#                 gae_lambda=self.gae_lambda,
#                 n_envs=self.n_envs, # Should be 1
#             )
#         # Need to know the internal batch size 'bs' of the environment
#         # This assumes the env instance has a 'batch_size' attribute
#         try:
#             self.env_batch_size = self.env.get_attr("batch_size")[0] # Get bs from the single env
#         except Exception:
#              # Fallback or error - requires env to expose batch_size
#              raise ValueError("PPO_custom requires the VecEnv environment to expose 'batch_size' via get_attr.")
#         self.actual_rollout_buffer_size = self.n_steps * self.env_batch_size # Total samples collected per rollout


#     # --- Need to override _setup_model if buffer size depends on env_batch_size ---
#     # Or ensure buffer is created correctly after env is available.
#     # Let's assume the __init__ buffer setup is okay for now, SB3 might resize.

#     def collect_rollouts(
#         self,
#         env: VecEnv,
#         callback: BaseCallback,
#         rollout_buffer: RolloutBuffer, # Should be DictRolloutBuffer instance
#         n_rollout_steps: int,
#     ) -> bool:
#         """
#         Collect experiences. Handles reshaping actions for the custom env.
#         """
#         assert self._last_obs is not None, "No previous observation was provided"
#         # Switch policy to evaluation mode
#         self.policy.set_training_mode(False)

#         n_steps = 0
#         rollout_buffer.reset() # Resets buffer for n_envs=1
#         if self.use_sde:
#             self.policy.reset_noise(self.env_batch_size) # Use bs for noise

#         callback.on_rollout_start()

#         while n_steps < n_rollout_steps: # n_rollout_steps is e.g., 128
#             if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
#                 self.policy.reset_noise(self.env_batch_size)

#             with torch.no_grad():
#                 # Convert observation to tensor. obs_as_tensor handles dicts.
#                 # self._last_obs has shape (n_envs, bs, ...) -> (1, bs, ...)
#                 obs_tensor = obs_as_tensor(self._last_obs, self.device)
#                 # Policy forward expects (1, bs, ...) and outputs flattened (bs,) actions, (bs, 1) values, (bs,) log_probs
#                 actions, values, log_probs = self.policy(obs_tensor) # shapes (bs,), (bs,1), (bs,)

#             # Action needs to be numpy array of shape (bs,) for the custom env's step
#             actions_np = actions.cpu().numpy() # Shape (bs,)

#             # Environment step expects (bs,) numpy array
#             new_obs, rewards, dones, infos = env.step(actions_np) # Shapes (bs,), (bs,), (bs,) for rewards, dones

#             # new_obs is dict with shapes (1, bs, ...) from VecEnv wrapper
#             # rewards, dones have shape (1,) from VecEnv wrapper, need to use bs internally?
#             # Let's assume VecEnv gives rewards/dones shape (1,) and we use the internal bs logic.
#             # If VecEnv gives (bs,), that's simpler. Let's check infos.
#             # infos is likely a list of dicts, length n_envs=1. infos[0] contains internal info.

#             # --- IMPORTANT: Adapt based on actual VecEnv output ---
#             # If VecEnv returns rewards/dones as shape (1,) representing the whole batch status (e.g., if ANY are done)
#             # this needs careful handling.
#             # Assuming VecEnv correctly returns rewards/dones per batch item, shape (bs,)?
#             # The provided env LogicEnv_gym_batch seems to return shapes like (bs,).
#             # The DummyVecEnv/SubprocVecEnv usually wrap this to (n_envs,).
#             # If using DummyVecEnv with n_envs=1, rewards/dones will be shape (1,). Need to extract.

#             # Let's assume DummyVecEnv wrapping: rewards/dones are (1,)
#             reward_val = rewards[0]
#             done_val = dones[0]
#             info_dict = infos[0] # Get info for the single environment

#             self.num_timesteps += self.env_batch_size # Increment by internal batch size

#             callback.update_locals(locals())
#             if not callback.on_step():
#                 return False

#             # _update_info_buffer expects a list/array of infos/dones for n_envs
#             self._update_info_buffer([info_dict] * self.env_batch_size, np.array([done_val] * self.env_batch_size)) # Expand done status
#             n_steps += 1 # One VecEnv step = one rollout step

#             # Handle timeout - needs adaptation if done_val represents the whole batch
#             if (
#                 done_val
#                 and info_dict.get("terminal_observation") is not None
#                 and info_dict.get("TimeLimit.truncated", False)
#             ):
#                  # Terminal obs needs potential batch handling if it contains bs items
#                  # Assuming info_dict["terminal_observation"] is the dict for the *entire* batch observation
#                  terminal_obs_dict = info_dict["terminal_observation"]
#                  # We need to predict values for the whole batch inside terminal_obs_dict
#                  # Shapes inside terminal_obs_dict are likely (bs, ...)
#                  # Need to wrap it for predict_values: add n_envs dim -> (1, bs, ...)
#                  terminal_obs_for_policy = {k: v.unsqueeze(0) for k,v in terminal_obs_dict.items()}
#                  terminal_obs_tensor = obs_as_tensor(terminal_obs_for_policy, self.device)

#                  with torch.no_grad():
#                      # predict_values outputs (n_envs*bs, 1) -> (bs, 1)
#                      terminal_values = self.policy.predict_values(terminal_obs_tensor) # Shape (bs, 1)

#                  # Add advantage to rewards - reward_val needs to represent the batch reward?
#                  # This is tricky. If reward_val is scalar, how to distribute terminal value?
#                  # Assuming rewards from env.step was shape (bs,) and VecEnv averaged it to (1,).
#                  # Let's revert to expecting rewards shape (bs,) directly.
#                  # *** Requires modifying VecEnv wrapper or assuming env returns (bs,) ***

#                  # --- Assuming env.step provides rewards/dones of shape (bs,) ---
#                  # And DummyVecEnv passes them as is inside a list/tuple of length 1? Let's assume direct access.
#                  # Rerun/debug needed if VecEnv aggregates rewards/dones.

#                  # Assuming rewards, dones passed to buffer are (bs,)
#                  if done_val: # Check if the whole VecEnv step is done (means internal done likely)
#                     # Need the actual terminal obs for the items that timed out
#                     # This requires info_dict to contain per-item terminal obs if mixing done/not done.
#                     # Simplification: If TimeLimit.truncated, apply to all? Risky.
#                     # TODO: Revisit timeout bootstrapping with batch envs

#                     # TEMPORARY Simplification: Add mean terminal value if timeout
#                     mean_terminal_value = terminal_values.mean()
#                     rewards = rewards + self.gamma * mean_terminal_value # Add to the whole batch rewards?

#             # Add to buffer - shapes must match buffer expectations
#             # Buffer expects (n_envs, ...) for obs, (n_envs,) for rewards/dones, (n_envs, ...) for actions? Check buffer add method.
#             # DictRolloutBuffer.add expects:
#             # obs: Dict[str, np.ndarray] shapes (n_envs, bs, ...) -> handled by _last_obs
#             # actions: np.ndarray shape (n_envs, *action_dim) -> Need (1, bs)
#             # rewards: np.ndarray shape (n_envs,) -> Need (1,)
#             # dones: np.ndarray shape (n_envs,) -> Need (1,)
#             # values: torch.Tensor shape (n_envs, 1) -> Need (1, 1) ? (Values are per batch item)
#             # log_probs: torch.Tensor shape (n_envs,) -> Need (1,)? (Log_probs are per batch item)

#             # Reshape outputs for buffer:
#             actions_for_buffer = actions.cpu().numpy().reshape(self.n_envs, self.env_batch_size) # -> (1, bs)
#             rewards_for_buffer = rewards.reshape(self.n_envs,) # Assumes rewards was (bs,) -> mean? No, sum? No. Needs VecEnv consistent output (1,)
#             dones_for_buffer = dones.reshape(self.n_envs,) # Assumes dones was (bs,) -> any? No. Needs VecEnv consistent output (1,)
#             # Values/log_probs are per item, need to average/sum for buffer? Or buffer handles batch?
#             # Let's average values/log_probs for the single VecEnv step
#             values_for_buffer = values.mean(dim=0, keepdim=True).reshape(self.n_envs, 1) # -> (1, 1)
#             log_probs_for_buffer = log_probs.mean(dim=0, keepdim=True).reshape(self.n_envs,) # -> (1,)


#             rollout_buffer.add(
#                  self._last_obs, # Dict observations, shape (1, bs, ...)
#                  actions_for_buffer, # (1, bs)
#                  rewards_for_buffer, # (1,) - Represents average/overall reward for the bs steps?
#                  self._last_episode_starts, # Should be (1,) matching dones_for_buffer
#                  values_for_buffer, # (1, 1) - Represents average value?
#                  log_probs_for_buffer, # (1,) - Represents average log_prob?
#              )
#             self._last_obs = new_obs # Update last obs, shape (1, bs, ...)
#             self._last_episode_starts = dones_for_buffer # Update done status for buffer, shape (1,)

#         # --- Compute returns and advantage ---
#         with torch.no_grad():
#             # Compute value for the last timestep for the whole batch
#             # new_obs shape (1, bs, ...)
#             # values shape (bs, 1)
#             values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
#             # Average value for the buffer's last_values, shape (1, 1)
#             last_values_for_buffer = values.mean(dim=0, keepdim=True).reshape(self.n_envs, 1)

#         # Buffer compute_returns_and_advantage expects last_values(n_envs, 1), dones(n_envs,)
#         rollout_buffer.compute_returns_and_advantage(last_values=last_values_for_buffer, dones=dones_for_buffer)

#         callback.on_rollout_end()

#         return True

#     def learn( # Keep learn mostly as is, relies on corrected collect_rollouts
#         self,
#         total_timesteps: int,
#         callback=None,
#         log_interval: int = 1,
#         tb_log_name: str = "PPO",
#         reset_num_timesteps: bool = True,
#         progress_bar=False,
#     ):
#         total_timesteps, callback = self._setup_learn(
#             total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar
#         )
#         callback.on_training_start(locals(), globals())

#         assert self.env is not None
#         iteration = 0

#         while self.num_timesteps < total_timesteps:
#             start_rollout = time.time()
#             print(f"Iteration {iteration+1}: Collecting rollouts...")
#             continue_training = self.collect_rollouts(
#                 self.env, callback, self.rollout_buffer, self.n_steps
#             )
#             print(f"Time to collect rollouts: {time.time() - start_rollout:.2f}s")

#             if not continue_training:
#                 break

#             iteration += 1
#             self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

#             # Display training infos
#             if log_interval is not None and iteration % log_interval == 0:
#                 fps = int(self.num_timesteps / (time.time() - self._start_time)) if (time.time() - self._start_time) > 0 else 0
#                 self.logger.record("time/iterations", iteration, exclude="tensorboard")
#                 if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
#                     self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
#                     self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
#                 self.logger.record("time/fps", fps)
#                 self.logger.record("time/time_elapsed", int(time.time() - self._start_time), exclude="tensorboard")
#                 self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
#                 self.logger.dump(step=self.num_timesteps)

#             # Train the model
#             print(f"Iteration {iteration}: Training model...")
#             start_train = time.time()
#             self.train()
#             print(f"Time to train: {time.time() - start_train:.2f}s")


#         callback.on_training_end()
#         return self

class PolicyNetwork(nn.Module):
    """ Actor network - Handles potential nested batch dims """
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=8, dropout_prob=0.2):
        super().__init__()
        self.embed_dim = embed_dim # Store embed_dim for reshaping later
        self.obs_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        self.out_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, obs_embeddings: torch.Tensor, action_embeddings: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_embeddings: Tensor shape (N, embed_dim) OR (n_envs, bs, embed_dim)
            action_embeddings: Tensor shape (N, padding_states, embed_dim) OR (n_envs, bs, padding_states, embed_dim)
            action_mask: Tensor shape (N, padding_states) OR (n_envs, bs, padding_states)

        Returns:
            logits: Tensor shape (N, padding_states)
        """
        # --- Determine combined batch size N and other dims, and flatten ---
        obs_shape = obs_embeddings.shape
        action_shape = action_embeddings.shape
        mask_shape = action_mask.shape

        if len(obs_shape) == 3: # Input is (n_envs, bs, embed_dim)
            n_envs, bs, _ = obs_shape
            N = n_envs * bs
            padding_states = action_shape[2]
            # Reshape inputs to (N, ...)
            obs_embeddings_flat = obs_embeddings.reshape(N, self.embed_dim)
            action_embeddings_flat = action_embeddings.reshape(N, padding_states, self.embed_dim)
            action_mask_flat = action_mask.reshape(N, padding_states)
        elif len(obs_shape) == 2: # Input is already flattened (N, embed_dim)
            N, _ = obs_shape
            padding_states = action_shape[1]
            obs_embeddings_flat = obs_embeddings
            action_embeddings_flat = action_embeddings
            action_mask_flat = action_mask
        else:
            raise ValueError(f"Unexpected shape for obs_embeddings: {obs_shape}")

        # --- Process Observation ---
        x = self.obs_transform(obs_embeddings_flat) # (N, hidden_dim)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        x = self.out_transform(x) # (N, embed_dim)

        # --- Prepare for BMM ---
        x_unsqueezed = x.unsqueeze(-1) # (N, embed_dim, 1)
        # action_embeddings_flat shape: (N, padding_states, embed_dim)

        # --- Perform BMM ---
        # Inputs: (N, padding_states, embed_dim) and (N, embed_dim, 1)
        # Output: (N, padding_states, 1)
        logits = torch.bmm(action_embeddings_flat, x_unsqueezed)
        logits = logits.squeeze(-1) # Shape: (N, padding_states)

        # --- Apply the action mask ---
        if action_mask_flat.shape != logits.shape:
             raise ValueError(f"Shape mismatch: action_mask_flat {action_mask_flat.shape}, logits {logits.shape}")

        # Ensure mask is boolean and apply it safely
        # Ensure mask is boolean. If it's 0/1 float/int, convert safely.
        if not action_mask_flat.dtype == torch.bool:
             action_mask_bool = action_mask_flat > 0.5 # Or appropriate threshold/conversion
        else:
             action_mask_bool = action_mask_flat

        # Apply mask: Set logits of invalid actions to negative infinity
        # Check device compatibility
        inf_tensor = torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
        logits = torch.where(action_mask_bool, logits, inf_tensor)


        return logits # Final shape (N, padding_states)

class ValueNetwork(nn.Module):
    """ Critic network - Handles potential nested batch dims """
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=8, dropout_prob=0.2):
        super().__init__()
        self.embed_dim = embed_dim # Store embed_dim for reshaping
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob)
        )
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_prob)
            ) for _ in range(num_layers)
        ])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, obs_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_embeddings: Tensor shape (N, embed_dim) OR (n_envs, bs, embed_dim)

        Returns:
            value: Tensor shape (N, 1)
        """
        # --- Reshape input if necessary ---
        obs_shape = obs_embeddings.shape
        if len(obs_shape) == 3: # Input is (n_envs, bs, embed_dim)
            N = obs_shape[0] * obs_shape[1]
            obs_embeddings_flat = obs_embeddings.reshape(N, self.embed_dim)
        elif len(obs_shape) == 2: # Input is already flattened (N, embed_dim)
            N = obs_shape[0]
            obs_embeddings_flat = obs_embeddings
        else:
            raise ValueError(f"Unexpected shape for obs_embeddings: {obs_shape}")

        # --- Process Observation ---
        x = self.input_layer(obs_embeddings_flat) # (N, hidden_dim)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        value = self.output_layer(x) # (N, 1)

        return value # Keep shape (N, 1) as expected by SB3 critics

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function integrating the custom embedder.
    Relies on internal Policy/Value networks to handle reshaping.
    """
    def __init__(
        self,
        last_layer_dim_pi: int, # Should match padding_states (action space size)
        last_layer_dim_vf: int = 1,
        embed_dim: int = 64,
        policy_kwargs: Optional[Dict] = None,
        value_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        policy_kwargs = policy_kwargs or {}
        value_kwargs = value_kwargs or {}

        # These represent the output dimensions *after* flattening of n_envs*bs
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # We need the embed_dim to pass to the sub-networks
        self.embed_dim = embed_dim

        self.policy_net = PolicyNetwork(embed_dim=self.embed_dim, **policy_kwargs)
        self.value_net = ValueNetwork(embed_dim=self.embed_dim, **value_kwargs)


    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass expects features tuple from CustomCombinedExtractor:
         - obs_embeddings: (n_envs, bs, embed_dim) or (N, embed_dim)
         - action_embeddings: (n_envs, bs, padding_states, embed_dim) or (N, padding_states, embed_dim)
         - action_mask: (n_envs, bs, padding_states) or (N, padding_states)

        Outputs:
         - action_logits: (N, padding_states)
         - state_values: (N, 1)
        """
        obs_embeddings, action_embeddings, action_mask = features
        # Pass features directly to sub-networks; reshaping happens inside them
        action_logits = self.policy_net(obs_embeddings, action_embeddings, action_mask)
        state_values = self.value_net(obs_embeddings)
        return action_logits, state_values

    def forward_actor(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ Forward pass for actor """
        obs_embeddings, action_embeddings, action_mask = features
        return self.policy_net(obs_embeddings, action_embeddings, action_mask)

    def forward_critic(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """ Forward pass for critic """
        # Critic typically only needs observation embedding
        obs_embeddings, _, _ = features
        return self.value_net(obs_embeddings)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor. Expects input tensors with shape (n_envs, bs, ...).
    Outputs features maintaining this structure for CustomNetwork to handle.
    """
    def __init__(self, observation_space: gym.spaces.Dict, embedder: Any, features_dim: int = 0):
        # features_dim=0 because we return a tuple, not a single flattened tensor
        super().__init__(observation_space, features_dim=features_dim)

        if embedder is None:
             raise ValueError("CustomCombinedExtractor requires an 'embedder' instance.")
        self.embedder = embedder
        # Assuming embedder has embed_dim attribute or similar
        try:
            self.embedding_dim = self.embedder.embed_dim
        except AttributeError:
            # Try to infer from sub-modules if necessary (adjust based on your embedder structure)
            try:
                self.embedding_dim = self.embedder.constant_embedder.embedding_dim
                print(f"NOTE: Inferred embed_dim={self.embedding_dim} from embedder's internal structure.")
            except AttributeError:
                 raise ValueError("Could not determine embedding_dim from the provided embedder.")

        # No network layers here, just extraction and embedding

    def forward(self, observations: PyTorchObs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input `observations` keys have tensors of shape (n_envs, bs, ...).
        Output tuple elements should preserve this (n_envs, bs, ...) structure.
        The subsequent CustomNetwork will handle the potential flattening.
        """
        # Shapes from buffer/obs_as_tensor likely (n_envs, bs, ...) or just (bs, ...) if n_envs=1
        # Let's verify the shapes received
        # print({k: v.shape for k, v in observations.items()}) # Useful for debugging

        obs_sub_indices = observations["state_sub_idx"]      # (n_envs, bs, padding_atoms, 3)
        action_sub_indices = observations["derived_sub_idx"] # (n_envs, bs, padding_states, padding_atoms, 3)
        action_mask = observations["action_mask"]            # (n_envs, bs, padding_states)

        # Ensure indices are long type for embedding lookups
        obs_sub_indices = obs_sub_indices.long()
        action_sub_indices = action_sub_indices.long()

        # Embedder's get_embeddings_batch should handle arbitrary leading dims.
        # Input (n_envs, bs, n_atoms, 3) -> Output (n_envs, bs, embed_dim)
        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices)
        # Input (n_envs, bs, n_states, n_atoms, 3) -> Output (n_envs, bs, n_states, embed_dim)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices)

        # Ensure action_mask keeps the leading dims: (n_envs, bs, padding_states)
        # It should already be this shape.

        # Return features preserving the structure for CustomNetwork
        return obs_embeddings, action_embeddings, action_mask

# --- CustomActorCriticPolicy ---
# Needs to handle the flattened outputs from CustomNetwork correctly,
# especially when creating the distribution.

class CustomActorCriticPolicy(ActorCriticPolicy): # Inherit from base ActorCriticPolicy
    """
    Custom Actor-Critic policy using the CustomCombinedExtractor and CustomNetwork.
    Handles Dict observation space.
    Relies on standard SB3 mechanisms for action distribution creation.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space, # Should be Dict space
        action_space: gym.spaces.Space,      # Should be Discrete space
        lr_schedule: Schedule, 
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None, # Not used by CustomNetwork directly
        activation_fn: Type[nn.Module] = nn.ReLU, # Passed to CustomNetwork if needed
        features_extractor_class: Type[BaseFeaturesExtractor] = CustomCombinedExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        # Add embed_dim explicitly if needed, otherwise inferred from extractor/kwargs
        embed_dim: Optional[int] = None,
        policy_net_kwargs: Optional[Dict] = None, # Specific args for PolicyNetwork
        value_net_kwargs: Optional[Dict] = None,  # Specific args for ValueNetwork
        *args,
        **kwargs,
    ):
        features_extractor_kwargs = features_extractor_kwargs or {}
        # Ensure the embedder is passed to the extractor
        if 'embedder' not in features_extractor_kwargs:
            raise ValueError("CustomActorCriticPolicy requires 'embedder' in features_extractor_kwargs.")

        # Determine embed_dim if not passed explicitly
        if embed_dim is None:
            try:
                # Assuming embedder has embed_dim attribute
                embed_dim = features_extractor_kwargs['embedder'].embed_dim
            except AttributeError:
                try:
                    # Try to infer from sub-modules (adjust based on your embedder structure)
                    embed_dim = features_extractor_kwargs['embedder'].constant_embedder.embedding_dim
                    print(f"NOTE: Inferred embed_dim={embed_dim} from embedder's internal structure for Policy.")
                except AttributeError:
                    raise ValueError("Cannot determine embed_dim. Pass it explicitly or ensure embedder has .embed_dim.")
        self.embed_dim = embed_dim # Store it

        # Store network kwargs
        self.policy_net_kwargs = policy_net_kwargs or {}
        self.value_net_kwargs = value_net_kwargs or {}

        # Note: features_dim=0 for extractor as it returns a tuple
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch, # Pass along, though CustomNetwork structure is mostly fixed
            activation_fn=activation_fn, # Pass along
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            # Set ortho_init to False, often better for custom/complex networks
            ortho_init=False,
            *args,
            **kwargs,
        )
        # self.action_dist is created in the super().__init__ call based on action_space
        # If action_space is Discrete, self.action_dist will be CategoricalDistribution


    def _build_mlp_extractor(self) -> None:
        """ Builds the CustomNetwork which acts as the MLP extractor """
        # Determine padding_states (action space size)
        if not isinstance(self.action_space, spaces.Discrete):
             # Raise error early if action space isn't Discrete, as the PolicyNetwork expects it
             raise ValueError(f"This CustomActorCriticPolicy requires a gym.spaces.Discrete action space, "
                              f"but got {type(self.action_space)}. Check your environment setup.")

        padding_states = self.action_space.n

        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=padding_states,       # Logits output dimension (size of Discrete space)
            last_layer_dim_vf=1,                    # Value output dimension
            embed_dim=self.embed_dim,               # Pass embed_dim
            policy_kwargs = self.policy_net_kwargs, # Pass specific kwargs
            value_kwargs = self.value_net_kwargs,   # Pass specific kwargs
        )

    # Override _get_action_dist_from_latent to use the standard SB3 method
    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes (logits).

        :param latent_pi: Latent code for the actor (logits).
                          Shape: (N, padding_states) where N is the combined batch size.
        :return: Action distribution instance.
        """
        # Use the policy's action distribution object (e.g., CategoricalDistribution instance)
        # to create the specific distribution instance for these logits.
        # This is the standard SB3 way.
        return self.action_dist.proba_distribution(action_logits=latent_pi)


    # Forward, evaluate_actions, predict_values can often remain as inherited
    # from MultiInputActorCriticPolicy, as they rely on the correctly implemented
    # _build_mlp_extractor and _get_action_dist_from_latent.
    # Double-check their implementation if issues arise.

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation (dictionary)
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
                - actions: (N,) where N is combined batch size
                - values: (N, 1)
                - log_prob: (N,)
        """
        # features is the tuple (obs_embeddings, action_embeddings, action_mask)
        # Shapes should be (n_envs, bs, ...) or potentially already flattened (N, ...)
        features = self.extract_features(obs)
        # latent_pi are logits (N, padding_states), latent_vf is value (N, 1)
        latent_pi, latent_vf = self.mlp_extractor(features)

        # Evaluate the values for the given observations
        values = latent_vf
        # action_distribution is created using self._get_action_dist_from_latent
        action_distribution = self._get_action_dist_from_latent(latent_pi)
        actions = action_distribution.get_actions(deterministic=deterministic)
        log_prob = action_distribution.log_prob(actions) # Log prob of chosen actions

        return actions, values, log_prob

    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy, given the observations.

        :param obs: Observation (dictionary)
        :param actions: Actions (Tensor shape (N,))
        :return: estimated value, log likelihood of taking those actions, entropy
                 - values: (N, 1)
                 - log_prob: (N,)
                 - entropy: (N,) or None
        """
        # features is the tuple (obs_embeddings, action_embeddings, action_mask)
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features) # Logits (N, pad_states), Values (N, 1)

        action_distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = action_distribution.log_prob(actions) # Log prob of the input actions
        values = latent_vf # Value estimate (N, 1)
        entropy = action_distribution.entropy()
        return values, log_prob, entropy # Entropy can be None depending on distribution

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation (dictionary)
        :return: the estimated values (Tensor shape (N, 1))
        """
        features = self.extract_features(obs) # Tuple
        # Critic calculation happens within mlp_extractor now
        _, values = self.mlp_extractor(features) # Use the value output directly
        # values = self.mlp_extractor.forward_critic(features) # Alternative if needed
        return values