"""
Simple test of PPO rollout with the batched environment.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from batched_env import BatchedVecEnv
from index_manager import IndexManager
from dataset import DataHandler
from ppo.ppo_rollout_custom import CustomRolloutCollector


class SimplePolicyNetwork(nn.Module):
    """Simple policy network for testing."""
    
    def __init__(self, hidden_size=64, num_actions=20):
        super().__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        
        # Simple MLP that takes flattened state and outputs logits
        # For simplicity, we'll just use random policy
        self.fc = nn.Linear(1, num_actions)  # Dummy input
    
    def forward(self, tensordict):
        """Forward pass that adds logits to tensordict."""
        batch_size = tensordict.batch_size[0]
        
        # Generate random logits (uniform random policy for testing)
        logits = torch.randn(batch_size, self.num_actions, device=tensordict.device)
        
        tensordict.set("logits", logits)
        return tensordict


class SimpleCriticNetwork(nn.Module):
    """Simple critic network for testing."""
    
    def __init__(self, hidden_size=64):
        super().__init__()
        self.fc = nn.Linear(1, 1)  # Dummy
    
    def forward(self, tensordict):
        """Forward pass that adds state_value to tensordict."""
        batch_size = tensordict.batch_size[0]
        
        # Generate dummy values
        values = torch.zeros(batch_size, 1, device=tensordict.device)
        
        tensordict.set("state_value", values)
        return tensordict


def test_ppo_rollout():
    """Test PPO rollout collector with batched environment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTesting PPO Rollout with CustomRolloutCollector")
    print(f"Device: {device}\n")
    
    # Load dataset
    print("Loading dataset...")
    data_handler = DataHandler(
        dataset_name='family',
        base_path='data',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt'
    )
    
    # Create index manager
    index_manager = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_vars=100,
        rules=data_handler.rules,
        device=device
    )
    
    # Setup environment
    batch_size = 4
    n_steps = 10
    queries = data_handler.train_queries[:batch_size * 3]  # Extra for multiple episodes
    labels = [1] * len(queries)
    query_depths = data_handler.train_queries_depths[:batch_size * 3]
    query_depths = [d if d is not None else 10 for d in query_depths]
    
    print(f"Creating environment with {batch_size} parallel envs")
    env = BatchedVecEnv(
        batch_size=batch_size,
        index_manager=index_manager,
        data_handler=data_handler,
        queries=queries,
        labels=labels,
        query_depths=query_depths,
        facts=set(data_handler.facts),
        mode='train',
        max_depth=10,
        memory_pruning=False,
        padding_atoms=10,
        padding_states=20,
        verbose=0,
        prover_verbose=0,
        device=device,
    )
    
    # Create simple policy and critic networks
    print("Creating policy and critic networks...")
    actor = SimplePolicyNetwork().to(device)
    critic = SimpleCriticNetwork().to(device)
    
    # Create rollout collector
    print(f"Creating rollout collector (n_steps={n_steps})...")
    collector = CustomRolloutCollector(
        env=env,
        actor=actor,
        n_envs=batch_size,
        n_steps=n_steps,
        device=device,
        debug=True
    )
    
    # Collect one rollout
    print("\nCollecting rollout...")
    experiences, stats = collector.collect(critic=critic)
    
    # Analyze results
    print("\n" + "=" * 80)
    print("ROLLOUT RESULTS")
    print("=" * 80)
    print(f"Number of experience steps: {len(experiences)}")
    print(f"Expected: {n_steps}")
    
    if len(experiences) > 0:
        sample_exp = experiences[0]
        print(f"\nSample experience keys: {sample_exp.keys()}")
        if 'next' in sample_exp.keys():
            print(f"Sample experience['next'] keys: {sample_exp['next'].keys()}")
        
        # Check rewards
        all_rewards = []
        for exp in experiences:
            if 'next' in exp.keys() and 'reward' in exp['next'].keys():
                rewards = exp['next']['reward']
                all_rewards.extend(rewards.cpu().flatten().tolist())
        
        print(f"\nTotal reward samples: {len(all_rewards)}")
        non_zero = [r for r in all_rewards if r != 0]
        print(f"Non-zero rewards: {len(non_zero)}")
        if non_zero:
            print(f"Non-zero values: {set(non_zero)}")
    
    # Episode info
    episode_infos = stats.get('episode_info', [])
    print(f"\nEpisode infos collected: {len(episode_infos)}")
    if episode_infos:
        print(f"Sample episode info: {episode_infos[0]}")
    
    print("=" * 80)
    
    # Cleanup
    collector.shutdown()
    
    print("\n✓ PPO rollout test completed successfully!")
    return True


if __name__ == '__main__':
    test_ppo_rollout()
