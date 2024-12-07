from typing import List, Union, Dict, Type, Optional, Callable, Tuple

from tensordict import TensorDict
import torch
from torch import nn
import gymnasium as gym

from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.distributions import (Distribution,make_proba_distribution)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

  
from utils import print_state_transition
from dataset import Rule
from environments.env_logic_gym import LogicEnv_gym
from stable_baselines3 import PPO

def eval_test(data: list[Rule], 
            labels: list[int],
            env: gym.Env,
            model: PPO, 
            deterministic: bool = True,
            verbose:int=0) -> Tuple[list[float], list[int]]:
    next_query = 0
    obs, _ = env.reset_from_query(data[next_query],labels[next_query])
    print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    rewards_list = []
    episode_len_list = []
    trajectory_reward = 0
    episode_len = 0
    while next_query < len(data)-1:
        print('query',next_query) if verbose >=1 else None
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, truncated, info = env.step(action)
        print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done'], action=env.tensordict['action'],truncated=truncated) if verbose >=1 else None
        trajectory_reward += rewards
        episode_len += 1
        if dones:
            next_query += 1
            obs, _ = env.reset_from_query(data[next_query],labels[next_query])
            rewards_list.append(trajectory_reward)
            episode_len_list.append(episode_len)
            trajectory_reward = 0
            episode_len = 0
            print_state_transition(env.tensordict['state'], env.tensordict['derived_states'],env.tensordict['reward'], env.tensordict['done']) if verbose >=1 else None
    return rewards_list, episode_len_list

class PolicyNetwork(nn.Module):
    def __init__(self, embed_dim=200):
        super().__init__()
        self.observation_transform = nn.Linear(embed_dim, embed_dim)

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

        # # return actions together with probs 
        # # pad actions to the len of probs to be able to stack them
        # print('actions', actions.shape, actions)
        # print('probs', probs.shape, probs)
        # actions = actions.unsqueeze(-1)
        # actions = torch.cat((actions, probs), dim=-1)        
        # print('actions', actions.shape, actions)
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
    