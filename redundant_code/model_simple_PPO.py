from redundant_code.env_logic_batched import BatchLogicProofEnv
from utils import print_td, print_rollout, get_max_arity, create_global_idx, read_embeddings, create_embed_tables
import janus_swi as janus

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


def transE_embedding(predicate_embeddings, constant_embeddings):
    """
    TransE function to compute atom embeddings.
    
    Arguments:
    - predicate_embeddings: Tensor of shape (batch_size, embedding_dim)
    - constant_embeddings: Tensor of shape (batch_size, 2, embedding_dim)
    
    Returns:
    - atom_embeddings: Tensor of shape (batch_size, embedding_dim)
    """
    # Separate the constants
    assert constant_embeddings.size(-2) == 2, "The second dimension of constant_embeddings should be 2 (arity)"
    assert predicate_embeddings.size(-2) == 1, "The second dimension of predicate_embeddings should be 1"
    predicate_embeddings = predicate_embeddings.squeeze(-2)
    constant_1 = constant_embeddings[..., 0, :]  # Shape: (256, 64)
    constant_2 = constant_embeddings[..., 1, :]  # Shape: (256, 64)
    # Compute the atom embedding using TransE formula
    atom_embeddings = predicate_embeddings + (constant_1 - constant_2)
    return atom_embeddings


        
class EmbeddingFunction:
    def __init__(self, constant_idx2emb=None, predicate_idx2emb=None, device="cpu"):
        """
        Initialize the embedding function.
        """
        self.device = device
        self.embed_dim = constant_idx2emb.size(1)
        self.constant_idx2emb = constant_idx2emb
        self.predicate_idx2emb = predicate_idx2emb
        # Set padding embedding (index 0) to zeros
        self.constant_idx2emb[0] = 0
        self.predicate_idx2emb[0] = 0
        
    def get_embeddings_batch(self, sub_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a batch of indices using vectorized operations.
        
        Args:
            indices: Tensor containing indices
            
        Returns:
            Tensor of embeddings with shape [..., embedding_dim]
        """
        # Not consider cache atom embedding for now
        # unique_indices = atom_indices.unique()
        # indices_in_dict = torch.tensor([idx in self.atom_idx2emb for idx in unique_indices])
        # precomputed_embeddings = torch.stack(
        #     [self.atom_idx2emb[idx.item()] for idx in unique_indices[indices_in_dict]]
        # )
        # computed_embeddings = torch.stack(
        #     [compute_embedding(idx.item()) for idx in unique_indices[~indices_in_dict]]
        # )

        # IF I GET "index out of range in self" ERROR, IT IS BECAUSE THE INDICES ARE OUT OF RANGE. ONCE I IMPLEMENT THE LOCAL INDICES, THIS WILL BE SOLVED
        # Look up the embeddings of predicates and args.
        # pred_arg_embeddings = F.embedding(sub_indices, self.embedding_table, padding_idx=0)
        predicate_indices = sub_indices[..., 0].unsqueeze(-1)
        constant_indices = sub_indices[..., 1:]
        predicate_embeddings = F.embedding(predicate_indices, self.predicate_idx2emb, padding_idx=0)
        constant_embeddings = F.embedding(constant_indices, self.constant_idx2emb, padding_idx=0)
        atom_embeddings = transE_embedding(predicate_embeddings, constant_embeddings)

        # # Sum pred & args embeddings to get atom embeddings.
        # pred_arg_embeddings = torch.cat([predicate_embeddings, constant_embeddings], dim=-2)
        # atom_embeddings = pred_arg_embeddings.sum(dim=-2)
        # Sum atom embeddings to get state embeddings.
        state_embeddings = atom_embeddings.sum(dim=-2)

        return state_embeddings
    
    def to(self, device):
        """Move embedding table to specified device."""
        self.device = device
        self.constant_idx2emb = self.constant_idx2emb.to(device)
        self.predicate_idx2emb = self.predicate_idx2emb.to(device)
        return self
        
    # @property
    # def weight(self):
    #     """Return the embedding table weights."""
    #     return self.embedding_table


class PolicyNetwork(nn.Module):
    def __init__(self, embedding_function):
        super().__init__()
        self.observation_transform = nn.Linear(embedding_function.embed_dim, embedding_function.embed_dim)
        self.embedding_dim = embedding_function.embed_dim
        self.embedding_function = embedding_function

    def format_indices(self, indices):
        """Ensures indices are in the correct shape for matrix operations."""
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        if indices.dim() == 1:
            indices = indices.unsqueeze(1)
        elif indices.dim() >= 2:
            indices = indices.unsqueeze(-3)
        return indices

    def forward(self, action_atom_indices, action_sub_indices, obs_sub_indices) -> TensorDict:
        # Prepare indices
        obs_sub_indices = self.format_indices(obs_sub_indices)

        # Obtain embeddings for observations and actions
        obs_embeddings = self.embedding_function.get_embeddings_batch(obs_sub_indices)
        action_embeddings = self.embedding_function.get_embeddings_batch(action_sub_indices)

        # Transform and calculate logits
        obs_features = self.observation_transform(obs_embeddings)
        logits = torch.matmul(obs_features, action_embeddings.transpose(-2, -1)).squeeze(-2)

        # Mask logits and compute probabilities
        logits = torch.where(action_atom_indices.sum(dim=-1) == 0, float('-inf'), logits)
        # logits = torch.where(action_atom_indices == 0, float('-inf'), logits)
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
        _, probs, _ = self.forward(tensordict["derived_atom_indices"], tensordict["derived_sub_indices"], tensordict["sub_index"])
        return torch.distributions.Categorical(probs=probs)
    
    def forward_dict(self, tensordict: TensorDict) -> TensorDict:
        """Generates actions and updates the TensorDict with probabilities and log-probabilities."""
        action, probs, sample_log_prob = self.forward(tensordict["derived_atom_indices"], tensordict["derived_sub_indices"], tensordict["sub_index"])
        tensordict.update({
            "action": action,
            "action_probs": probs,
            "sample_log_prob": sample_log_prob
        })
        return tensordict



class ValueNetwork(nn.Module):
    def __init__(self, embedding_function):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_function.embed_dim, embedding_function.embed_dim),
            nn.ReLU(),
            nn.Linear(embedding_function.embed_dim, 1))
        self.embedding_function = embedding_function
        
    def forward(self, sub_indices) -> TensorDict:
        value = self.network(self.embedding_function.get_embeddings_batch(sub_indices))
        value = value.squeeze(-1)
        return value


    
def simple_rollout(env: BatchLogicProofEnv, policy: PolicyNetwork = None, batch_size: int=2, steps: int=10, tensordict: TensorDict = None) -> TensorDict:
    ''' CAREFUL!!! pytroch doesnt stack the keys that are lists properly (for the tensors it should be fine). OR maybe it is because of the data_spec. Check it out'''
    data = []
    if tensordict is None:
        _data = env.reset(env.gen_params(batch_size=[batch_size]))
    else:
        _data = env.reset(tensordict)
    for i in range(steps):
        # print('i', i,'------------------------------------')
        _data["action"] = env.action_spec.sample() if policy is None else policy.forward_dict(_data)["action"]
        _data = env.step(_data)

        # for state, action, derived_states,reward,done in zip(_data['state'], _data['action'],_data['derived_states'],_data['reward'], _data['done']):
        #     print(*state, '-> action', action.item(),'/', len(derived_states)-1)
        #     print('reward',reward)
        #     print('Done',done)
        #     print('     Derived states:',*derived_states,'\n')
        
        # print('actions',_data['action'],'rewards',_data['reward'],'dones',_data['done'])
        data.append(_data) # We append it here because we want to keep the "next" data. Those will be datapoint samples
        if _data["done"].all():
            # print('\nDONE',_data["done"])
            break
        _data = step_mdp(_data, keep_other=True,exclude_reward=False,exclude_done=False,exclude_action=False)

    data = TensorDict.stack(data, dim=1)
    return data

 
def simplified_ppo_train(env,policy_module, value_module, 
                         n_epochs=100, n_episodes=100000, batch_size=32, n_rollout=10,
                         clip_ratio=0.2, lr=3e-4, gamma=0.99, gae_lambda=0.95, knowledge_f=None, test_f=None, max_arity=1):
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

    policy_optimizer = torch.optim.Adam(policy_module.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_module.parameters(), lr=lr)

    for episode in range(n_episodes):
        # COLLECT DATA
        init_td = env.gen_params(batch_size=batch_size)
        env.reset_atom_var()
        data = simple_rollout(env,policy=policy_module.module,steps=n_rollout,tensordict=init_td)
        # data = simple_rollout(env, steps=n_rollout, tensordict=init_td)
        # FORWARD: CALCULATE ADVANTAGES (VALUES) AND POLICY
        print('\n')
        for epoch in range(n_epochs):
            data = advantage_module(data)
            # print('data after advantage',data)
            # data = policy_module.forward_dict(data)

            # PPO LOSS UPDATE
            loss = clip_ppo_loss(data)
            # print('loss objective from loop',loss['loss_objective'])
            # Update Policy Network
            policy_optimizer.zero_grad()
            loss["loss_objective"].backward()
            policy_optimizer.step()

            # Update Value Network
            value_optimizer.zero_grad()
            loss["loss_critic"].backward()
            value_optimizer.step()


            # the reward is the last reward of the rollout for each query
            reward = data['reward'][:,-1].mean()


            # LOGGING
            if (epoch + 1) % 100 == 0:
                print(f"Episode {episode}, Epoch {epoch + 1}",end=", ")
                print(f"Average Reward: {reward.item():.3f}",end=", ")
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


#knowledge_f = "data/ancestor.pl"
#test_f = None,

# knowledge_f = "data/countries_s1_train.pl"
# test_f = "data/countries_s1_test.pl"
# constant_embed_f = "data/countries_s1/constant_embeddings.pkl"
# predicate_embed_f = "data/countries_s1/predicate_embeddings.pkl"

knowledge_f = "data/s2_designed/train.pl"
test_f = "data/s2_designed/test.pl"
constant_embed_f = "data/s2_designed/constant_embeddings.pkl"
predicate_embed_f = "data/s2_designed/predicate_embeddings.pkl"


janus.consult(knowledge_f)
max_arity = get_max_arity(knowledge_f)
constant_str2idx, predicate_str2idx = create_global_idx(knowledge_f)
constant_no = max(constant_str2idx.values())
predicate_no = max(predicate_str2idx.values())
constant_idx2emb, predicate_idx2emb = read_embeddings(constant_embed_f, predicate_embed_f, constant_str2idx, predicate_str2idx)
# TODO: current fixed vars, if dynamic, need to create dynamically and may add entry to tensor_dict
VARIABLE_NO = 500
constant_idx2emb, predicate_idx2emb = create_embed_tables(constant_idx2emb, predicate_idx2emb, VARIABLE_NO)


# Training configuration
config = {
    "n_epochs": 2000,
    "n_episodes": 100000,
    "batch_size": 1,
    "n_rollout": 50,
    "clip_ratio": 0.2,
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}

env = BatchLogicProofEnv(batch_size=config["batch_size"], knowledge_f=knowledge_f, test_f=test_f, max_arity=max_arity, constant_str2idx=constant_str2idx, predicate_str2idx=predicate_str2idx, constant_no=constant_no, predicate_no=predicate_no, variable_no=VARIABLE_NO)
embedding_func = EmbeddingFunction(constant_idx2emb, predicate_idx2emb)
policy_net = PolicyNetwork(embedding_func)
value_net = ValueNetwork(embedding_func)

value_module = TensorDictModule(
                    value_net,
                    in_keys=["sub_index"],
                    out_keys=["value"]
                )

policy_module = TensorDictModule(
                    policy_net,
                    in_keys=["derived_atom_indices", "derived_sub_indices","sub_index"],
                    out_keys=["action", "action_probs", "sample_log_prob"]
                )



if __name__ == "__main__":


    # # ----------------------Test the environment, policy and value net----------------------
    # init_td = env.reset(env.gen_params(batch_size=config["batch_size"]))
    # # td = env.rollout(100,tensordict=init_td,
    # #                  policy=policy_net,
    # #                  auto_reset = False,
    # #                  break_when_any_done = False,
    # #                  )
    # value_net.forward(init_td["sub_index"])
    # td = simple_rollout(env,policy=policy_net,steps=config["n_rollout"],tensordict=init_td)
    # print('reward',td['reward'])
    # # print('rollout',td)
    # # print_rollout(td)
    # # print_td(td, exclude_states=True)
    # # ---------------------------------------------------------------------------------------------



    # ----------------------Train the model----------------------
    simplified_ppo_train(env, policy_module, value_module, **config)
    # -----------------------------------------------------------

    # rollout = env.rollout(3, break_when_any_done=False, policy=policy_module)
    # print(rollout)

    # frames_per_batch = 1000
    # total_frames = 100000
    # device = "cpu"
    #
    # collector = SyncDataCollector(
    #     env,
    #     policy_module,
    #     frames_per_batch=frames_per_batch,
    #     total_frames=total_frames,
    #     split_trajs=False,
    #     device=device,
    # )
    # for i, tensors in enumerate(collector):
    #     print(i, tensors)
    #     if i > 10:
    #         break


    # # NOT SUPPORTED YET
    # policy_module = ProbabilisticActor(
    #     # spec=env.action_spec,
    #     module=policy_module,
    #     in_keys=["derived_indices","index"],
    #     out_keys=["action", "action_probs", "sample_log_prob"],
    #     distribution_class=Categorical,  # Specify the distribution class
    #     return_log_prob=True,
    # )


