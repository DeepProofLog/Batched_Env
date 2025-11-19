import sys
import os
import torch
from types import SimpleNamespace

# Add path
sys.path.insert(0, os.getcwd())

from test_envs.test_engine_sb3 import setup_sb3_engine
from test_envs.test_env_sb3 import setup_sb3_env, test_sb3_env_single_query
from test_envs.test_env_tensor import setup_tensor_env, test_tensor_env_batched
from test_envs.test_all_configs import compare_traces_deterministic

def reproduce():
    query_tuple = ('aunt', '905', '912')
    split = 'train'
    
    # Config
    cfg = SimpleNamespace(
        dataset="family",
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        skip_unary_actions=True,
        end_proof_action=False,
        memory_pruning=True,
        use_exact_memory=True,
        verbose=2, # Enable verbose to see debug prints
        prover_verbose=0,
        device='cpu',
        deterministic=True,
        max_depth=20,
        seed=42,
        max_total_runtime_vars=1000
    )

    print("Setting up SB3 Env...")
    sb3_env, im_str, dh_str = setup_sb3_env(dataset="family", config=cfg)
    
    print("Running SB3 Query...")
    sb3_result = test_sb3_env_single_query(query_tuple, (sb3_env, im_str, dh_str), split=split, verbose=True)
    
    print("\nSetting up Batched Tensor Env...")
    # For batched, we need to pass batch_size=1
    batched_env, debug_helper, constant_no, im_batched, dh_batched = setup_tensor_env(
        dataset="family", 
        batch_size=1, 
        config=cfg
    )
    
    print("Running Batched Tensor Query...")
    # We need to construct the input for test_tensor_env_batched
    # It expects a list of queries
    queries = [(split, query_tuple)]
    
    # We can't easily use test_tensor_env_batched because it sets up its own env.
    # We should use the internal logic or just call test_tensor_env_batched directly.
    # test_tensor_env_batched calls setup_tensor_env internally.
    
    batched_result = test_tensor_env_batched(queries, (batched_env, debug_helper, constant_no, im_batched, dh_batched), cfg)
    
    # Compare
    traces = {
        'sb3_env': [sb3_result],
        'batched_tensor_env': batched_result['traces']
    }
    
    compare_traces_deterministic(traces, verbose=True)

if __name__ == "__main__":
    reproduce()
