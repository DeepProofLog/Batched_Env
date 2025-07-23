import math
import time
from typing import List, Union, Dict, Type, Optional, Callable, Tuple

import torch as th
import torch
from torch import log, nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import (Distribution,make_proba_distribution)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv 
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import obs_as_tensor
from stable_baselines3.common.utils import safe_mean



# class that takes PPO but modifies the collect_rollouts method to return the actions and probs
class PPO_custom(PPO):
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

            with th.no_grad():
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
                    with th.no_grad():
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

        with th.no_grad():
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
    
    def forward(self, obs_embeddings, action_embeddings, action_atom_indices):
        # Process observation embeddings through initial transformation
        x = self.obs_transform(obs_embeddings)
        # Pass through a series of residual blocks to deepen the representation
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual  # Residual connection for improved gradient flow
        # Project back to the original embedding space
        x = self.out_transform(x)
        # Compute similarity (dot product) between observation and action embeddings
        logits = torch.matmul(x, action_embeddings.transpose(-2, -1)).squeeze(-2)
        # logits = F.cosine_similarity(x, action_embeddings, dim=-1) # Compare along the embedding dimension
        padding_condition = action_atom_indices.sum(dim=-1).sum(dim=-1) 
        logits = torch.where(padding_condition == 0, float('-inf'), logits)  

        return logits

class ValueNetwork(nn.Module):
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
    
    def forward(self, obs_embeddings):
        # Process observation embeddings through the input layer
        x = self.input_layer(obs_embeddings)
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
        obs_embeddings, action_embeddings, action_atom_indices = features
        probs = self.policy_network(obs_embeddings, action_embeddings, action_atom_indices) # (batch_size=n_envs,pad_states)
        value = self.value_network(obs_embeddings) # (batch_size=n_envs,n_states=1)
        return probs, value
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the actor network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent policy representation.
        """
        obs_embeddings, action_embeddings, action_atom_indices, = features
        return self.policy_network(obs_embeddings, action_embeddings, action_atom_indices)


    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the critic network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent value representation.
        """
        obs_embeddings, _, _ = features
        return self.value_network(obs_embeddings)



class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,*args,**kwargs):
        super().__init__(observation_space, features_dim=1)

        self.embedder = kwargs.pop("embedder", None)
        if self.embedder is not None:
            self.embedding_dim = self.embedder.embed_dim

    
    def format_indices(self, indices):
        """Ensures indices are in the correct shape for matmul operations
        Args:
            indices: torch.Tensor of shape (batch_size=n_envs,1,1,pad_atoms,3)
            2nd dim is to match the shape of derived_sub_indices
            3rd dim is to match the shape of derived_atom_indices
        Returns:
            torch.Tensor of shape (batch_size=n_envs,1,1,pad_atoms,3)
        """
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        elif indices.dim() >= 2:
            indices = indices.unsqueeze(-3)
        return indices

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from observations including valid action mask."""

        obs_sub_indices = observations["sub_index"] # (batch_size=n_envs,1,pad_atoms,3) 2nd dim is to match the shape of derived_sub_indices 
        action_sub_indices = observations["derived_sub_indices"] # (batch_size=n_envs,pad_states,pad_atoms,3) 
        action_atom_indices = observations["derived_sub_indices"] # (batch_size=n_envs,pad_states,pad_atoms,3)

        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices.long()) # (batch_size=n_envs,n_states=1,embedding_dim)
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices.long()) # (batch_size=n_envs,pad_states,embedding_dim)

        return obs_embeddings, action_embeddings, action_atom_indices #, valid_actions_mask



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
        self.kge_inference_engine = kwargs.pop("kge_inference_engine", None)
        self.index_manager = kwargs.pop("index_manager", None)
        self.features_extractor_kwargs = kwargs.pop('features_extractor_kwargs', {})

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

    def _build_mlp_extractor(self) -> None:
        """
        Builds the extractor using the CustomNetwork that retains TorchRL logic.
        """
        self.mlp_extractor = CustomNetwork(
            last_layer_dim_pi=self.features_extractor._features_dim,
            last_layer_dim_vf= 1, # self.features_extractor._features_dim,
            embed_dim=self.features_extractor_kwargs['features_dim']
        )

    def get_kge_log_probs(self, obs: PyTorchObs, actions: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
        actions_squeezed = actions.squeeze(1) if actions.ndim > 1 else actions
        batch_indices = torch.arange(actions_squeezed.shape[0], device=actions_squeezed.device)

        chosen_action_sub_indices = obs["derived_sub_indices"][batch_indices, actions_squeezed]
        chosen_action_pred_indices = chosen_action_sub_indices[:, 0, 0]

        kge_action_mask = torch.isin(chosen_action_pred_indices, self.kge_indices_tensor.to(chosen_action_pred_indices.device))
        kge_batch_indices = kge_action_mask.nonzero(as_tuple=False).squeeze(-1)

        if kge_batch_indices.numel() > 0:
            atoms_to_predict, original_indices = [], []

            for batch_idx in kge_batch_indices:
                kge_action_sub_index = chosen_action_sub_indices[batch_idx, 0, :]
                kge_action_str = self.index_manager.subindex_to_str(kge_action_sub_index)
                kge_pred_str = self.index_manager.predicate_idx2str.get(kge_action_sub_index[0].item())

                if kge_action_str and kge_pred_str:
                    original_pred_str = kge_pred_str.removesuffix('_kge')
                    original_atom_str = f"{original_pred_str}{kge_action_str[len(kge_pred_str):]}"
                    atoms_to_predict.append(original_atom_str)
                    # convert tensor → python int once, it makes indexing simpler
                    original_indices.append(int(batch_idx))
        
            if atoms_to_predict:
                # 1. KGE scores
                scores = self.kge_inference_engine.predict_batch(atoms_to_predict)
                # print(f"Atoms to predict: {len(atoms_to_predict)}, {atoms_to_predict}, Scores: {scores}")
                kge_log_probs = torch.log(
                    torch.as_tensor(scores, device=log_prob.device, dtype=log_prob.dtype)
                    + 1e-9
                )

                # 2. Normalise shapes so we can safely scatter-update
                flat_lp = log_prob.view(-1)          # works for () , (B,) or (B,1)
                idx_tensor = torch.tensor(original_indices,
                                        device=log_prob.device, dtype=torch.long)
                flat_lp[idx_tensor] = kge_log_probs  # in-place
                log_prob = flat_lp.view_as(log_prob) # restore original shape

        return log_prob

    def forward(self, obs: torch.Tensor, deterministic: bool = False, verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features) # (batch_size=n_envs,n_states=1,embedding_dim)
            latent_vf = self.mlp_extractor.forward_critic(vf_features) # (batch_size=n_envs,n_states=1)

        # Get value for the state
        values = latent_vf # (batch_size=n_envs,n_states=1)
        # Get action probabilities and create distribution
        action_logits = latent_pi # (batch_size=n_envs,pad_states)

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Sample a single action from the distribution
        actions = distribution.get_actions(deterministic=deterministic)
        # Get log probability of the chosen action
        log_prob = distribution.log_prob(actions)

        # If a KGE action was taken, overwrite its log_prob with the KGE score
        if self.kge_inference_engine is not None:
            log_prob = self.get_kge_log_probs(obs, actions, log_prob)
        return actions, values, log_prob


    def evaluate_actions(self, obs: PyTorchObs, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Create action distribution using logits
        action_logits = latent_pi
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Get log probability of the given actions
        log_prob = distribution.log_prob(actions)
        values = latent_vf
        entropy = distribution.entropy()

        # If a KGE action was taken, overwrite its log_prob with the KGE score
        if self.kge_inference_engine is not None:
            log_prob = self.get_kge_log_probs(obs, actions, log_prob)
        
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        values = latent_vf
        return values

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        action_logits = latent_pi
        # print('number of eval actions', action_logits.shape, action_logits)
        return self.action_dist.proba_distribution(action_logits=action_logits)