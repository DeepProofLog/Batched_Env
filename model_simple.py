from environments.env_logic import BatchLogicProofEnv
from utils import print_td, print_rollout

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import multiprocessing

from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.objectives.value import GAE
from torchrl.objectives.ppo import PPOLoss
from torchrl.objectives import ClipPPOLoss





        
class EmbeddingFunction:
    def __init__(self, num_embeddings: int=1000000, embedding_dim: int = 64, device="cpu"):
        """
        Initialize the embedding function.
        
        Args:
            num_embeddings: Maximum number of embeddings to support
            embedding_dim: Dimension of each embedding vector
            device: Device to store embeddings on
        """
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize embedding table all at once
        self.embedding_table = torch.randn(num_embeddings, embedding_dim, device=device)
        # Set padding embedding (index 0) to zeros
        self.embedding_table[0] = 0
        
    def get_embeddings_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of indices using vectorized operations.
        
        Args:
            indices: Tensor containing indices
            
        Returns:
            Tensor of embeddings with shape [..., embedding_dim]
        """
        # IF I GET "index out of range in self" ERROR, IT IS BECAUSE THE INDICES ARE OUT OF RANGE. ONCE I IMPLEMENT THE LOCAL INDICES, THIS WILL BE SOLVED
        return torch.embedding(self.embedding_table, indices, padding_idx=0)
    
    def to(self, device):
        """Move embedding table to specified device."""
        self.device = device
        self.embedding_table = self.embedding_table.to(device)
        return self
        
    @property
    def weight(self):
        """Return the embedding table weights."""
        return self.embedding_table


class PolicyNetwork(nn.Module):
    def __init__(self, embedding_function, embedding_dim: int = 64):
        super().__init__()
        self.observation_transform = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.embedding_function = embedding_function

    def format_indices(self, indices):
        """Ensures indices are in the correct shape for matrix operations."""
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        elif indices.dim() >= 2:
            indices = indices.unsqueeze(-1)
        return indices

    def forward(self, action_indices, obs_indices) -> TensorDict:
        # Prepare indices
        obs_indices = self.format_indices(obs_indices)

        # Obtain embeddings for observations and actions
        obs_embeddings = self.embedding_function.get_embeddings_batch(obs_indices)
        action_embeddings = self.embedding_function.get_embeddings_batch(action_indices)

        # Transform and calculate logits
        obs_features = self.observation_transform(obs_embeddings)
        logits = torch.matmul(obs_features, action_embeddings.transpose(-2, -1)).squeeze(-2)

        # Mask logits and compute probabilities
        logits = torch.where(action_indices == 0, float('-inf'), logits)
        probs = F.softmax(logits, dim=-1)

        # PROBABILISTIC:Sample action with probabilities given by probs
        # Sample actions and calculate log probabilities
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        sample_log_prob = dist.log_prob(action).detach()

        # # DETERMINISTIC: Choose action with highest probability
        # action = torch.argmax(probs, dim=-1)
        return action,probs,sample_log_prob

    def get_dist(self, tensordict: TensorDict) -> torch.distributions.Categorical:
        """Returns the action distribution based on the current policy."""
        _, probs, _ = self.forward(tensordict["derived_indices"], tensordict["index"])
        return torch.distributions.Categorical(probs=probs)
    
    def forward_dict(self, tensordict: TensorDict) -> TensorDict:
        """Generates actions and updates the TensorDict with probabilities and log-probabilities."""
        action, probs, sample_log_prob = self.forward(tensordict["derived_indices"], tensordict["index"])
        tensordict.update({
            "action": action,
            "action_probs": probs,
            "sample_log_prob": sample_log_prob
        })
        return tensordict



class ValueNetwork(nn.Module):
    def __init__(self, embedding_function, embedding_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.embedding_function = embedding_function
        
    def forward(self, indices) -> TensorDict:
        value = self.network(self.embedding_function.get_embeddings_batch(indices))
        value = value.squeeze(-1)
        return value


    
def simple_rollout(env: BatchLogicProofEnv, policy: PolicyNetwork = None, batch_size: int=2, steps: int=3, tensordict: TensorDict = None) -> TensorDict:
    ''' CAREFUL!!! pytroch doesnt stack the keys that are lists properly (for the tensors it should be fine). OR maybe it is because of the data_spec. Check it out
    TO IMPLEMENT: STOP ONLY WHEN ALL THE BATCHES ARE DONE OR STEPS IS REACHED'''
    data = []
    if tensordict is None:
        _data = env.reset(env.gen_params(batch_size=[batch_size]))
    else:
        _data = env.reset(tensordict)
    # print_td(_data)
    for i in range(steps):
        # print('i', i,'------------------------------------')
        _data["action"] = env.action_spec.sample() # Random action
        # _data =  policy.forward_dict(_data) # action taken from polic
        _data = env.step(_data)
        # print_td(_data)
        data.append(_data) # We append it here because we want to keep the "next" data. Those will be datapoint samples
        if _data["done"].all():
            break
        _data = step_mdp(_data, keep_other=True,exclude_reward=False,exclude_done=False,exclude_action=False)

    data = TensorDict.stack(data, dim=1)
    return data

 
def simplified_ppo_train(env,policy_module, value_module, 
                         n_epochs=100, batch_size=32, n_rollout=10,
                         clip_ratio=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95):
    """Main PPO training loop"""

    clip_ppo_loss = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_ratio,
        entropy_bonus=True,
        entropy_coef=0.01,
        critic_coef=1.0,
    )
    clip_ppo_loss.set_keys(value="value",value_target="returns")

    advantage_module = GAE(   
        gamma=gamma, 
        lmbda=gae_lambda, 
        value_network=value_module, #We calculate it by hand
        average_gae=True
    )
    advantage_module.set_keys(value="value",value_target="returns")

    for epoch in range(n_epochs):
        # COLLECT DATA
        init_td = env.gen_params(batch_size=batch_size)
        data = simple_rollout(env,steps=3,tensordict=init_td)
        
        # FORWARD: CALCULATE ADVANTAGES (VALUES) AND POLICY
        data = advantage_module(data)
        # print('data after advantage',data)
        data = policy_module.forward_dict(data)

        # PPO LOSS UPDATE
        loss = clip_ppo_loss(data)

        # LOGGING
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}",end=", ")
            print(f"Average Reward: {data['reward'].mean().item():.3f}",end=", ")
            print(f"Policy Loss: {loss['loss_objective'].item():.3f}",end=", ")
            print(f"Value Loss: {loss['loss_critic'].item():.3f}",end=", ")
            print(f"Entropy Loss: {loss['loss_entropy'].item():.3f}",end=", ")

            # print(f"kl_approx: {loss['kl_approx'].item():.3f}",end=", ")
            # print(f"ESS: {loss['ESS'].item():.3f}",end=", ")
            # print(f"clip_fraction: {loss['clip_fraction'].item():.3f}",end=", ")
            # print(f"Entropy: {loss['entropy'].item():.3f}",end=", ")
            
            # print(f"Total Loss: {loss.item():.3f}")
            print("-" * 50)
    
    return None

# Training configuration
config = {
    "n_epochs": 10000,
    "batch_size": 32,
    "n_rollout": 10,
    "clip_ratio": 0.2,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95
}

env = BatchLogicProofEnv(batch_size=config["batch_size"])
policy_net = PolicyNetwork(EmbeddingFunction())
value_net = ValueNetwork(EmbeddingFunction())

value_module = TensorDictModule(
                    value_net,
                    in_keys=["index"],
                    out_keys=["value"]
                )

policy_module = TensorDictModule(
                    policy_net,
                    in_keys=["derived_indices","index"],
                    out_keys=["action", "action_probs", "sample_log_prob"]
                )

simplified_ppo_train(env, policy_module, value_module, **config)






# # NOT SUPPORTED YET
# policy_module = ProbabilisticActor(
#     # spec=env.action_spec,
#     module=policy_module,
#     in_keys=["derived_indices","index"],
#     out_keys=["action", "action_probs", "sample_log_prob"],
#     distribution_class=Categorical,  # Specify the distribution class
#     return_log_prob=True,
# )





# This is just to test the environment and the policy----------------------
batch_size = 2
env = BatchLogicProofEnv(batch_size=batch_size)
policy_net = PolicyNetwork(EmbeddingFunction())

init_td = env.reset(env.gen_params(batch_size=batch_size))
# td = env.rollout(100,tensordict=init_td,
#                  policy=policy_net,
#                  auto_reset = False,
#                  break_when_any_done = False,
#                  )
td = simple_rollout(env,steps=3,tensordict=init_td)
print('rollout',td)
# print_rollout(td)
# print_td(td, exclude_states=True)


