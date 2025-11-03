"""
Incremental test for parallel environments - adding components one by one.

This test incrementally adds project components to identify which one causes
multiprocessing issues with 8 parallel environments:

1. Level 0: Pure dummy env (baseline - should work)
2. Level 1: + DataHandler 
3. Level 2: + IndexManager
4. Level 3: + NegativeSampler
5. Level 4: + Full environment

Each level is tested separately to pinpoint where the problem occurs.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from types import SimpleNamespace
from torchrl.envs import EnvBase, ParallelEnv
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict


# ============================================================================
# LEVEL 0: Pure Dummy Environment (Baseline)
# ============================================================================

class Level0_PureDummyEnv(EnvBase):
    """Pure dummy environment with no project dependencies"""
    def __init__(self, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.step_count = 0
        
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(4,)),
            shape=()
        )
        self.action_spec = DiscreteTensorSpec(n=2, shape=())
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        self.step_count = 0
        return TensorDict({
            'observation': torch.randn(4),
            'done': torch.tensor([False]),
        }, batch_size=())
    
    def _step(self, tensordict):
        self.step_count += 1
        done = self.step_count >= 10
        return TensorDict({
            'observation': torch.randn(4),
            'reward': torch.randn(1),
            'done': torch.tensor([done]),
        }, batch_size=())
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed


# ============================================================================
# LEVEL 1: + DataHandler
# ============================================================================

class Level1_WithDataHandler(EnvBase):
    """Environment that uses DataHandler"""
    def __init__(self, data_handler=None, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.step_count = 0
        
        # Store data handler (test if it can be pickled)
        self.data_handler = data_handler
        
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(4,)),
            shape=()
        )
        self.action_spec = DiscreteTensorSpec(n=2, shape=())
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        self.step_count = 0
        # Access data handler to verify it works
        if self.data_handler:
            _ = len(self.data_handler.train_queries)
        return TensorDict({
            'observation': torch.randn(4),
            'done': torch.tensor([False]),
        }, batch_size=())
    
    def _step(self, tensordict):
        self.step_count += 1
        done = self.step_count >= 10
        return TensorDict({
            'observation': torch.randn(4),
            'reward': torch.randn(1),
            'done': torch.tensor([done]),
        }, batch_size=())
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed


# ============================================================================
# LEVEL 2: + IndexManager
# ============================================================================

class Level2_WithIndexManager(EnvBase):
    """Environment that uses DataHandler + IndexManager"""
    def __init__(self, data_handler=None, index_manager=None, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.step_count = 0
        
        self.data_handler = data_handler
        self.index_manager = index_manager
        
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(4,)),
            shape=()
        )
        self.action_spec = DiscreteTensorSpec(n=2, shape=())
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        self.step_count = 0
        # Access both components
        if self.data_handler:
            _ = len(self.data_handler.train_queries)
        if self.index_manager:
            _ = len(self.index_manager.constants)
        return TensorDict({
            'observation': torch.randn(4),
            'done': torch.tensor([False]),
        }, batch_size=())
    
    def _step(self, tensordict):
        self.step_count += 1
        done = self.step_count >= 10
        return TensorDict({
            'observation': torch.randn(4),
            'reward': torch.randn(1),
            'done': torch.tensor([done]),
        }, batch_size=())
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed


# ============================================================================
# LEVEL 3: + NegativeSampler
# ============================================================================

class Level3_WithSampler(EnvBase):
    """Environment that uses DataHandler + IndexManager + NegativeSampler"""
    def __init__(self, data_handler=None, index_manager=None, env_id=0):
        super().__init__()
        self.env_id = env_id
        self.step_count = 0
        
        self.data_handler = data_handler
        self.index_manager = index_manager
        
        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(shape=(4,)),
            shape=()
        )
        self.action_spec = DiscreteTensorSpec(n=2, shape=())
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,))
        self.done_spec = DiscreteTensorSpec(n=2, shape=(1,), dtype=torch.bool)
        
    def _reset(self, tensordict=None, **kwargs):
        self.step_count = 0
        # Access all components including sampler
        if self.data_handler:
            _ = len(self.data_handler.train_queries)
            if hasattr(self.data_handler, 'sampler') and self.data_handler.sampler:
                _ = self.data_handler.sampler
        if self.index_manager:
            _ = len(self.index_manager.constants)
        return TensorDict({
            'observation': torch.randn(4),
            'done': torch.tensor([False]),
        }, batch_size=())
    
    def _step(self, tensordict):
        self.step_count += 1
        done = self.step_count >= 10
        return TensorDict({
            'observation': torch.randn(4),
            'reward': torch.randn(1),
            'done': torch.tensor([done]),
        }, batch_size=())
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        return seed


# ============================================================================
# Test Functions
# ============================================================================

def test_level(level_name, env_class, env_kwargs, n_envs=8, n_steps=5):
    """Test a specific level with given environment class and kwargs"""
    print(f"\n{'='*70}")
    print(f"Testing {level_name}")
    print(f"{'='*70}")
    
    try:
        # Create environment factory
        def create_env():
            return env_class(**env_kwargs)
        
        print(f"Creating ParallelEnv with {n_envs} workers...")
        env = ParallelEnv(
            num_workers=n_envs,
            create_env_fn=create_env,
        )
        
        print(f"Calling reset()...")
        td = env.reset()
        print(f"✓ Reset successful - batch_size: {td.batch_size}")
        
        print(f"Running {n_steps} steps...")
        for step in range(n_steps):
            action_td = TensorDict(
                {'action': torch.zeros(n_envs, dtype=torch.long)},
                batch_size=torch.Size([n_envs])
            )
            td = env.step(action_td)
            if step == 0:
                print(f"✓ Step {step+1} successful - batch_size: {td.batch_size}")
        
        print(f"✓ All {n_steps} steps completed")
        
        env.close()
        print(f"✓ Environment closed")
        print(f"\n{'='*70}")
        print(f"✅ {level_name} PASSED")
        print(f"{'='*70}")
        return True
        
    except Exception as e:
        import traceback
        print(f"\n{'='*70}")
        print(f"❌ {level_name} FAILED")
        print(f"{'='*70}")
        print(f"Error: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all incremental tests"""
    print("\n" + "="*70)
    print("INCREMENTAL PARALLEL ENVIRONMENT TEST")
    print("="*70)
    print("\nThis test adds components incrementally to identify issues:")
    print("  Level 0: Pure dummy environment (baseline)")
    print("  Level 1: + DataHandler")
    print("  Level 2: + IndexManager")
    print("  Level 3: + NegativeSampler")
    print("="*70)
    
    results = {}
    
    # Level 0: Baseline
    results['Level 0'] = test_level(
        "Level 0: Pure Dummy Environment",
        Level0_PureDummyEnv,
        {'env_id': 0}
    )
    
    if not results['Level 0']:
        print("\n⚠️  Baseline test failed - there's a fundamental issue")
        return results
    
    # Level 1: + DataHandler
    print("\nLoading DataHandler for Level 1...")
    try:
        from dataset import DataHandler
        
        dh = DataHandler(
            dataset_name='countries_s3',
            base_path='data',
            janus_file=None,
            train_file='train.txt',
            valid_file='valid.txt',
            test_file='test.txt',
            rules_file='rules.txt',
            facts_file='train.txt',
            n_eval_queries=16,
            corruption_mode='dynamic',
        )
        print(f"✓ DataHandler loaded: {len(dh.train_queries)} train queries")
        
        results['Level 1'] = test_level(
            "Level 1: With DataHandler",
            Level1_WithDataHandler,
            {'data_handler': dh, 'env_id': 0}
        )
    except Exception as e:
        print(f"❌ Failed to load DataHandler: {e}")
        import traceback
        traceback.print_exc()
        results['Level 1'] = False
    
    if not results['Level 1']:
        print("\n⚠️  Level 1 failed - issue is with DataHandler pickling/multiprocessing")
        return results
    
    # Level 2: + IndexManager
    print("\nLoading IndexManager for Level 2...")
    try:
        from index_manager import IndexManager
        
        im = IndexManager(
            constants=dh.constants,
            predicates=dh.predicates,
            max_total_vars=100,
            rules=dh.rules,
            padding_atoms=6,
            max_arity=dh.max_arity,
            device='cpu',
        )
        im.build_fact_index(dh.facts)
        print(f"✓ IndexManager loaded: {len(im.constants)} constants")
        
        results['Level 2'] = test_level(
            "Level 2: With DataHandler + IndexManager",
            Level2_WithIndexManager,
            {'data_handler': dh, 'index_manager': im, 'env_id': 0}
        )
    except Exception as e:
        print(f"❌ Failed to load IndexManager: {e}")
        import traceback
        traceback.print_exc()
        results['Level 2'] = False
    
    if not results['Level 2']:
        print("\n⚠️  Level 2 failed - issue is with IndexManager pickling/multiprocessing")
        return results
    
    # Level 3: + NegativeSampler
    print("\nLoading NegativeSampler for Level 3...")
    try:
        from neg_sampling import get_sampler
        
        dh.sampler = get_sampler(
            data_handler=dh,
            index_manager=im,
            corruption_scheme=['head', 'tail'],
            device='cpu',
        )
        print(f"✓ NegativeSampler loaded")
        
        results['Level 3'] = test_level(
            "Level 3: With DataHandler + IndexManager + NegativeSampler",
            Level3_WithSampler,
            {'data_handler': dh, 'index_manager': im, 'env_id': 0}
        )
    except Exception as e:
        print(f"❌ Failed to load NegativeSampler: {e}")
        import traceback
        traceback.print_exc()
        results['Level 3'] = False
    
    if not results['Level 3']:
        print("\n⚠️  Level 3 failed - issue is with NegativeSampler pickling/multiprocessing")
        return results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    for level, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{level}: {status}")
    print("="*70)
    
    # Find first failure
    failed_levels = [level for level, passed in results.items() if not passed]
    if failed_levels:
        print(f"\n⚠️  First failure: {failed_levels[0]}")
        print(f"The issue is likely in the component(s) added at this level.")
    else:
        print("\n✅ All levels passed!")
    
    return results


if __name__ == '__main__':
    import multiprocessing as mp
    # Ensure we're using spawn (default on most systems, but explicit is better)
    mp.set_start_method('spawn', force=True)
    
    results = run_all_tests()
