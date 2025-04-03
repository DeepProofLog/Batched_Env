from typing import List, Union, Dict, Type, Optional, Callable, Tuple

from tensordict import TensorDict
import numpy as np  
import torch as th
import torch
from torch import nn
from torch.nn.functional import softmax
import torch.nn.functional as F

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
            
            if n_steps % (n_rollout_steps // 5) == 0:
                print(f"Collecting rollouts: {n_steps}/{n_rollout_steps} steps")

            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert observations to tensor and pass to policy
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # print('*'*50)
                # print('\nobs_tensor')
                # for key, value in obs_tensor.items():
                #     print(key,value.shape,'\n',value)
                actions, values, log_probs = self.policy(obs_tensor)
                # print('actions', actions.shape, actions)
                # print('\n','*'*50)
            actions = actions.cpu().numpy()

            # Execute actions in environment
            new_obs, rewards, dones, infos = env.step(actions)
            # print('n_steps', n_steps)
            # print('observations', [(k,v.shape) for k,v in self._last_obs.items()])
            # print('values', values)
            # print('actions', actions)
            # print('log_probs', log_probs)
            # print('dones', dones)
            # print only rewards where done is True
            # print('rewards done', [r for r,d in zip(rewards,dones) if d])
            # if any(dones):
            #     print('ratio of positive done rewards', sum([r for r,d in zip(rewards,dones) if d]) / len([r for r,d in zip(rewards,dones) if d]))
            # print('-'*100)
            # print('sum, length of rewards', sum(rewards), len(rewards))
            # print('dones', dones)
            # print('infos', infos)
            # print('length of infos',len(infos))
            # for info in infos:
            #     for k,v in info.items():
            #         print('info4: ',k,v) if k != 'terminal_observation' else None
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            
            self._update_info_buffer(infos, dones)
            n_steps += 1

            # print('len, avg, ep_info_buffer rewards', len(self.ep_info_buffer), np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]),[ep_info["r"] for ep_info in self.ep_info_buffer])
            # print('len, avg, ep_info_buffer lengths', len(self.ep_info_buffer), np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]),[ep_info["l"] for ep_info in self.ep_info_buffer])
           
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
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    print('Rewards modified in rollout buffer because of terminal state!!!!!!!')
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        # Record rollout metrics for logging
        # print('len, avg, ep_info_buffer rewards', len(self.ep_info_buffer), np.mean([ep_info["r"] for ep_info in self.ep_info_buffer]),[ep_info["r"] for ep_info in self.ep_info_buffer])
        # print('_len, avg, ep_info_buffer lengths', len(self.ep_info_buffer), np.mean([ep_info["l"] for ep_info in self.ep_info_buffer]),[ep_info["l"] for ep_info in self.ep_info_buffer])
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
        print('Embedding dim in policy', embed_dim)
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
        # Mask out invalid actions: where the sum over action_atom_indices is 0, set logits to -inf
        logits = torch.where(action_atom_indices.sum(dim=-1) == 0, float('-inf'), logits)
        return logits

class ValueNetwork(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=8, dropout_prob=0.2):
        print('Embedding dim in value', embed_dim)
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

# class PolicyNetwork(nn.Module):
#     def __init__(self, embed_dim=200):
#         super().__init__()

#         # self.observation_transform = nn.Linear(embed_dim, embed_dim)
#         self.observation_transform = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#         )

#     def forward(self, obs_embeddings, action_embeddings, action_atom_indices) -> TensorDict:
#     # def forward(self, obs_embeddings, action_embeddings, action_atom_indices, valid_actions_mask) -> TensorDict:
#         """
#         Calculate logits for actions given observation embeddings and action embeddings.
        
#         Args:
#             obs_embeddings: Embedded observation (batch_size=n_envs,n_states=1,embedding_dim)
#             action_embeddings: Embedded possible actions (batch_size=n_envs,pad_states,embedding_dim)
#             action_atom_indices: Indices of atoms in actions (batch_size=n_envs,pad_states,pad_atoms)
#             valid_actions_mask: Boolean mask indicating which actions are valid (batch_size=n_envs,pad_states)
            
#         Returns:
#             Logits for actions, with -inf for invalid actions. (batch_size=n_envs,pad_states)
#         """
#         # Transform observation features
#         obs_features = self.observation_transform(obs_embeddings)
#         # Calculate similarity between observation and action embeddings
#         logits = torch.matmul(obs_features, action_embeddings.transpose(-2, -1)).squeeze(-2) # (batch_size=n_envs,pad_states)
#         logits = torch.where(action_atom_indices.sum(dim=-1) == 0, float('-inf'), logits)
#         return logits


# class ValueNetwork(nn.Module):
#     def __init__(self, embed_dim=200):
#         super().__init__()

#         # self.network = nn.Sequential(
#         #     nn.Linear(embed_dim, embed_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embed_dim, 1))

#         self.network = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embed_dim),    # Normalize activations to stabilize learning
#             nn.Dropout(0.1),            # Add a bit of dropout to prevent overfitting
#             nn.Linear(embed_dim, embed_dim),  # Maintain the dimensionality for deeper representation
#             nn.ReLU(),
#             nn.LayerNorm(embed_dim),
#             nn.Dropout(0.1),
#             nn.Linear(embed_dim, 1)      # Final output remains a single value
#         )

#     def forward(self, obs_embeddings) -> TensorDict:
#         value = self.network(obs_embeddings)
#         value = value.squeeze(-1)
#         return value


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
        # obs_embeddings, action_embeddings, action_atom_indices, valid_actions_mask = features
        probs = self.policy_network(obs_embeddings, action_embeddings, action_atom_indices) # (batch_size=n_envs,pad_states)
        # probs = self.policy_network(obs_embeddings, action_embeddings, action_atom_indices, valid_actions_mask) # (batch_size=n_envs,pad_states)
        value = self.value_network(obs_embeddings) # (batch_size=n_envs,n_states=1)
        return probs, value
    
    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the actor network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent policy representation.
        """
        obs_embeddings, action_embeddings, action_atom_indices, = features
        return self.policy_network(obs_embeddings, action_embeddings, action_atom_indices)
        # obs_embeddings, action_embeddings, action_atom_indices, valid_actions_mask = features
        # return self.policy_network(obs_embeddings, action_embeddings, action_atom_indices, valid_actions_mask)


    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the critic network.
        Accepts features (which can include sub_indices for embedding) and outputs the latent value representation.
        """
        obs_embeddings, _, _ = features
        # obs_embeddings, _, _, _ = features
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
        
        # Obtain embeddings for observations and actions
        obs_sub_indices = observations["sub_index"] # (batch_size=n_envs,1,pad_atoms,3) 2nd dim is to match the shape of derived_sub_indices 
        action_sub_indices = observations["derived_sub_indices"] # (batch_size=n_envs,pad_states,pad_atoms,3) 
        action_atom_indices = observations["derived_atom_indices"] # (batch_size=n_envs,pad_states,pad_atoms) 
        # # Always get the valid_actions_mask if available
        # valid_actions_mask = observations.get("valid_actions_mask", None) # shape (batch_size=n_envs,pad_states)
        
        # obs_sub_indices = self.format_indices(obs_sub_indices) # (batch_size=n_envs,1,1,pad_atoms,3) #3rd dim is for matmul with predicate embeddings
        
        # Generate embeddings
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
        # num actions is given by action_logits where it is not -inf
        num_actions = torch.sum(action_logits != float('-inf'), dim=-1)
        # print('num_actions', num_actions) 
        # show the non -inf logits
        # print('logits', action_logits[action_logits != float('-inf')])
        # print the ratio of actions that are more than one
        # print('ratio of actions more than one', sum(num_actions > 1) / len(num_actions))
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        
        # Sample a single action from the distribution
        actions = distribution.get_actions(deterministic=deterministic)
        # Get log probability of the chosen action
        log_prob = distribution.log_prob(actions)
        print('*'*100) if verbose else None
        print('values', values.shape) if verbose else None
        print('actions_logits', action_logits.shape) if verbose else None
        print('distribution params', distribution.distribution.probs.shape) if verbose else None
        print('action chosen', actions.shape, actions) if verbose else None
        print('log_prob of action chosen', log_prob.shape, log_prob) if verbose else None
        print('*'*100) if verbose else None
        # print('log_prob', log_prob)
        # print('num_actions', actions)
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
        return self.action_dist.proba_distribution(action_logits=action_logits)
    