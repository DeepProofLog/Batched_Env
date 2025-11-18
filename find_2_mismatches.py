"""Find the 2 mismatches in default settings"""
import sys
sys.path.insert(0, ".")

from test_envs.test_all_configs import Configs, prepare_queries, run_config, compare_traces_deterministic
from types import SimpleNamespace

# Default config
cfg = SimpleNamespace(
    dataset="family",
    n_queries=400,
    seed=42,
    memory_pruning=True,
    skip_unary_actions=False,
    end_proof_action=False,
    endt_action=False,
    endf_action=False,
    kge_action=False,
    canonical_action_order=True,
    engine="python",
    padding_atoms=6,
    padding_states=40,
    verbose=0,
    max_depth=20,
    use_exact_memory=False,  # Using BloomFilter
    deterministic=True,
)

# Get queries
queries = prepare_queries(dataset=cfg.dataset, n_queries=cfg.n_queries, seed=cfg.seed)

# Run sb3_env
print("Running sb3_env...")
sb3_config = Configs.get_config_by_name('sb3_env')
sb3_results = run_config(sb3_config, queries, cfg)

# Run batched_tensor_env
print("Running batched_tensor_env...")
tensor_config = Configs.get_config_by_name('batched_tensor_env')
tensor_results = run_config(tensor_config, queries, cfg)

# Compare with verbose
traces_dict = {
    'sb3_env': sb3_results['traces'],
    'batched_tensor_env': tensor_results['traces']
}

print("\n" + "="*80)
print("DETAILED COMPARISON")
print("="*80)

# Find mismatches manually
mismatches = []
for q_idx in range(len(queries)):
    sb3_trace = sb3_results['traces'][q_idx]
    tensor_trace = tensor_results['traces'][q_idx]
    
    if sb3_trace['success'] != tensor_trace['success']:
        query_idx, (pred, head, tail) = queries[q_idx]
        mismatches.append({
            'idx': q_idx,
            'query': (pred, head, tail),
            'sb3_success': sb3_trace['success'],
            'tensor_success': tensor_trace['success'],
            'sb3_steps': len(sb3_trace['trace']),
            'tensor_steps': len(tensor_trace['trace'])
        })
    elif len(sb3_trace['trace']) != len(tensor_trace['trace']):
        query_idx, (pred, head, tail) = queries[q_idx]
        mismatches.append({
            'idx': q_idx,
            'query': (pred, head, tail),
            'sb3_success': sb3_trace['success'],
            'tensor_success': tensor_trace['success'],
            'sb3_steps': len(sb3_trace['trace']),
            'tensor_steps': len(tensor_trace['trace']),
            'type': 'trace_length'
        })
    else:
        # Check step by step
        for step_idx, (sb3_step, tensor_step) in enumerate(zip(sb3_trace['trace'], tensor_trace['trace'])):
            if sb3_step['num_actions'] != tensor_step['num_actions']:
                query_idx, (pred, head, tail) = queries[q_idx]
                mismatches.append({
                    'idx': q_idx,
                    'query': (pred, head, tail),
                    'step': step_idx,
                    'type': 'num_actions',
                    'sb3_actions': sb3_step['num_actions'],
                    'tensor_actions': tensor_step['num_actions']
                })
                break

print(f"Found {len(mismatches)} mismatches:")
for m in mismatches:
    print(f"\nQuery {m['idx']}: {m['query']}")
    if 'step' in m:
        print(f"  Mismatch at step {m['step']}")
        print(f"  Type: {m.get('type', 'unknown')}")
        if m.get('type') == 'num_actions':
            print(f"  sb3_env actions: {m['sb3_actions']}")
            print(f"  tensor_env actions: {m['tensor_actions']}")
    else:
        print(f"  sb3_env: {m['sb3_success']} in {m['sb3_steps']} steps")
        print(f"  tensor_env: {m['tensor_success']} in {m['tensor_steps']} steps")
