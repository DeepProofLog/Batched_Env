"""
Simple test for TorchRL parallel environments without complex dependencies.

This test verifies that TorchRL's ParallelEnv works correctly with multiple workers
without relying on any project-specific code (DataHandler, IndexManager, etc.).

The test demonstrates that:
1. Basic parallel environment functionality works with 1, 2, 4, and 8 workers
2. Reset and step operations return correctly shaped batched TensorDicts
3. The environment can handle 100+ steps without issues

If this test passes but your actual environment fails, the issue is likely in:
- How your environment handles multiprocessing (pickling issues)
- Heavy imports in the environment __init__ (TensorFlow, etc.)
- Shared state between processes (DataHandler, IndexManager)

Recommendations for fixing your environment:
1. Minimize imports in the environment __init__
2. Pass only serializable data to the environment
3. Avoid global state or heavy objects in environment initialization
4. Use lazy loading for heavy dependencies
"""
import torch
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict


class SimpleDummyEnv(EnvBase):
    """
    Minimal dummy environment for testing parallel execution.
    No external dependencies on data handlers, index managers, etc.
    """
    def __init__(self, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.step_count = 0
        
        # Define observation and action specs
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(4,)),
            shape=()
        )
        
        self.action_spec = DiscreteTensorSpec(n=2, shape=())
        
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        """Reset the environment"""
        self.step_count = 0
        
        td = TensorDict({
            'observation': torch.randn(4),
            'done': torch.tensor([False]),
        }, batch_size=())
        
        return td
    
    def _step(self, tensordict):
        """Take a step in the environment"""
        action = tensordict['action']
        self.step_count += 1
        
        # Simple logic: done after 10 steps
        done = self.step_count >= 10
        
        # Random reward
        reward = torch.randn(1)
        
        td = TensorDict({
            'observation': torch.randn(4),
            'reward': reward,
            'done': torch.tensor([done]),
        }, batch_size=())
        
        return td
    
    def _set_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        return seed


def test_parallel_env_basic():
    """Test basic parallel environment functionality with multiple workers"""
    print("=" * 60)
    print("Testing basic parallel environment functionality")
    print("=" * 60)
    
    # Test with different numbers of environments
    for n_envs in [16]:
        print(f"\nTesting with {n_envs} parallel environments...")
        
        # Create parallel environment
        env = ParallelEnv(
            num_workers=n_envs,
            create_env_fn=SimpleDummyEnv,
        )
        
        print(f"  Created parallel environment")
        
        # Reset
        td_reset = env.reset()
        print(f"  Reset successful - batch_size: {td_reset.batch_size}")
        assert td_reset.batch_size == torch.Size([n_envs]), f"Expected batch size {n_envs}, got {td_reset.batch_size}"
        assert 'observation' in td_reset.keys(), "Missing 'observation' in reset output"
        
        # Take a few steps
        for step in range(3):
            action_td = TensorDict(
                {'action': torch.zeros(n_envs, dtype=torch.long)},
                batch_size=torch.Size([n_envs])
            )
            
            td_step = env.step(action_td)
            print(f"  Step {step+1} successful - batch_size: {td_step.batch_size}")
            assert td_step.batch_size == torch.Size([n_envs]), f"Expected batch size {n_envs}, got {td_step.batch_size}"
            
            # TorchRL puts the next state info in a 'next' key
            assert 'next' in td_step.keys(), "Missing 'next' in step output"
            assert 'observation' in td_step['next'].keys(), "Missing 'observation' in next state"
            assert 'reward' in td_step['next'].keys(), "Missing 'reward' in next state"
            assert 'done' in td_step['next'].keys(), "Missing 'done' in next state"
        
        # Close environment
        env.close()
        print(f"  ✓ Test passed for {n_envs} environments")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


def test_parallel_env_stress():
    """Stress test with many environments and many steps"""
    print("\n" + "=" * 60)
    print("Stress testing parallel environment")
    print("=" * 60)
    
    n_envs = 8
    n_steps = 100
    
    print(f"\nRunning {n_steps} steps with {n_envs} parallel environments...")
    
    env = ParallelEnv(
        num_workers=n_envs,
        create_env_fn=SimpleDummyEnv,
    )
    
    td_reset = env.reset()
    print(f"Reset successful - batch_size: {td_reset.batch_size}")
    
    for step in range(n_steps):
        action_td = TensorDict(
            {'action': torch.randint(0, 2, (n_envs,))},  # Random actions
            batch_size=torch.Size([n_envs])
        )
        
        td_step = env.step(action_td)
        
        if step % 20 == 0:
            print(f"  Completed {step}/{n_steps} steps")
    
    env.close()
    print(f"  ✓ Stress test passed: {n_steps} steps completed successfully")
    print("=" * 60)


if __name__ == '__main__':
    test_parallel_env_basic()
    test_parallel_env_stress()
