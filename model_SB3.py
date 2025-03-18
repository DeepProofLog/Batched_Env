from typing import List, Union, Dict, Type, Optional, Callable, Tuple

from tensordict import TensorDict
import numpy as np  
import torch as th
import torch
from torch import nn
from torch.nn.functional import softmax

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

import time


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

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(actions)
                else:
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            self._update_info_buffer(infos, dones)

            if not callback.on_step():
                return False

            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if done and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Record rollout metrics here, before on_rollout_end
        if self.ep_info_buffer and len(self.ep_info_buffer) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))

        callback.on_rollout_end()  # Now SB3ModelCheckpoint can see "rollout/ep_rew_mean"
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
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps
            )
            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)
            super().train()

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
    def __init__(self, embed_dim=200):
        super().__init__()

        self.observation_transform = nn.Linear(embed_dim, embed_dim)

        # self.observation_transform = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim))

        # hidden_dim = int(embed_dim * 1.5)
        # dropout_prob = 0.1
        # self.observation_transform = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(hidden_dim, embed_dim)
        # )

    def forward(self, obs_embeddings, action_embeddings, action_atom_indices) -> TensorDict:
        # Transform and calculate logits
        obs_features = self.observation_transform(obs_embeddings)
        logits = torch.matmul(obs_features, action_embeddings.transpose(-2, -1)).squeeze(-2)
        # Mask logits and compute probabilities
        logits = torch.where(action_atom_indices.sum(dim=-1) == 0, float('-inf'), logits)
        # probs = F.softmax(logits, dim=-1)
        probs = logits
        return probs


class ValueNetwork(nn.Module):
    def __init__(self, embed_dim=200):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1))

        # hidden_dim = int(embed_dim * 1.5)
        # dropout_prob = 0.1
        # self.network = nn.Sequential(
        #     nn.Linear(embed_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, 1)
        # )
        
    def forward(self, obs_embeddings) -> TensorDict:
        value = self.network(obs_embeddings)
        value = value.squeeze(-1)
        return value


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
        last_layer_dim_pi: int = 64,  # TO CHANGE
        last_layer_dim_vf: int = 1,  # TO CHANGE
        embed_dim: int = 200
    ):
        super(CustomNetwork, self).__init__()
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # TorchRL-style Policy and Value components
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
        probs = self.policy_network(obs_embeddings, action_embeddings, action_atom_indices)  
        value = self.value_network(obs_embeddings) 
        return probs, value
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the actor network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent policy representation.
        """
        obs_embeddings, action_embeddings, action_atom_indices = features
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
        """Ensures indices are in the correct shape for matrix operations."""
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        elif indices.dim() >= 2:
            indices = indices.unsqueeze(-3)
        return indices

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        # Obtain embeddings for observations and actions
        obs_sub_indices = observations["sub_index"]
        action_sub_indices = observations["derived_sub_indices"]
        action_atom_indices = observations["derived_atom_indices"]

        # print('obs_sub_indices:',obs_sub_indices.shape, list(obs_sub_indices[:3,0,:].numpy()))
        # print('action_sub_indices:',action_sub_indices.shape, list(action_sub_indices[0,:3,0,:].numpy()))
        # print('action_atom_indices:',action_atom_indices.shape, list(action_atom_indices[0,:3,:3].numpy()),'\n')


        obs_sub_indices = self.format_indices(obs_sub_indices)

        obs_embeddings = self.embedder.get_embeddings_batch(obs_sub_indices.long())
        action_embeddings = self.embedder.get_embeddings_batch(action_sub_indices.long())
        
        return obs_embeddings, action_embeddings, action_atom_indices



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

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = latent_vf

        probs = latent_pi
        distribution = self.action_dist.proba_distribution(action_logits=probs)
        # distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic) # new sampled probs
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  

        # probs_norm = softmax(probs, dim=-1)
        # print('\nprobs', probs.shape, probs)
        # print('probs normalized', probs_norm.shape, probs_norm)
        # print('logprobs from probs normalized', torch.log(probs_norm))
        # print('logprobs from dist', distribution.log_prob( torch.tensor([i for i in range(0,probs.shape[-1])]) ))
        # print('action (sampled from distr)', actions.shape, actions)
        # print('log_prob of that action', log_prob.shape, log_prob)
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

        # distribution = self._get_action_dist_from_latent(latent_pi)
        probs = latent_pi
        distribution = self.action_dist.proba_distribution(action_logits=probs)
        log_prob = distribution.log_prob(actions)
        # values = self.value_net(latent_vf)
        values = latent_vf
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        # features = super().extract_features(obs, self.vf_features_extractor)
        # latent_vf = self.mlp_extractor.forward_critic(features)
        # return self.value_net(latent_vf)
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
        probs = latent_pi
        # return self._get_action_dist_from_latent(latent_pi)
        return self.action_dist.proba_distribution(action_logits=probs)
    