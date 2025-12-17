"""
Environment Compiled Parity Tests.

Tests verifying that EvalOnlyEnvOptimized (compiled environment) produces
EXACTLY the same step-by-step traces as the tensor BatchedEnv when both use
skip_unary_actions=False.

Usage:
    # Run with pytest
    pytest tests/parity/test_env_compiled_parity.py -v -s
    
    # Run with torch.compile enabled
    pytest tests/parity/test_env_compiled_parity.py -v -s --compile
    
    # CLI
    python tests/parity/test_env_compiled_parity.py --dataset countries_s3 --n-queries 50
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Dict, Tuple, Optional
import random

import pytest
import torch
import numpy as np
from tensordict import TensorDict

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
TEST_ENVS_ROOT = ROOT / "tests" / "other" / "test_envs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TEST_ENVS_ROOT) not in sys.path:
    sys.path.insert(1, str(TEST_ENVS_ROOT))

from data_handler import DataHandler
from index_manager import IndexManager
from tensor.tensor_unification import UnificationEngine
from unification import UnificationEngineVectorized
from tensor.tensor_env import BatchedEnv
from env import EnvVec
import tensor.tensor_utils.tensor_utils as utils_funcs


# ============================================================================
# Configuration
# ============================================================================

def create_default_config() -> SimpleNamespace:
    """Default parameters for parity tests matching test_env_parity.py."""
    return SimpleNamespace(
        # Dataset/query selection
        dataset="countries_s3",
        n_queries=200,
        data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'),
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
        corruption_mode="dynamic",
        
        # Environment behavior
        max_depth=20,
        padding_atoms=6,
        padding_states=40,
        max_derived_per_state=40,
        skip_unary_actions=False,  # Must be False for parity
        end_proof_action=True,
        memory_pruning=True,
        use_exact_memory=True,  # Must be True for parity - compiled env uses exact hash comparison
        reward_type=0,
        prover_verbose=0,
        max_total_runtime_vars=1000,
        device="cpu",
        
        # Seeds / logging
        seed=42,
        verbose=False,
        debug=False,
        batch_size=100,  # Process queries in batches for efficiency
    )


def clone_config(config: SimpleNamespace) -> SimpleNamespace:
    """Create a shallow clone of a SimpleNamespace."""
    return SimpleNamespace(**vars(config))


# ============================================================================
# Query Preparation
# ============================================================================

def prepare_queries(config: SimpleNamespace) -> List[Tuple[str, Tuple[str, str, str]]]:
    """Prepare a shuffled list of queries for the requested dataset."""
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
    )
    
    all_queries = []
    for q in dh.train_queries:
        all_queries.append(('train', (q.predicate, q.args[0], q.args[1])))
    for q in dh.valid_queries:
        all_queries.append(('valid', (q.predicate, q.args[0], q.args[1])))
    for q in dh.test_queries:
        all_queries.append(('test', (q.predicate, q.args[0], q.args[1])))
    
    rng = random.Random(config.seed)
    rng.shuffle(all_queries)
    
    if config.n_queries is not None:
        all_queries = all_queries[:config.n_queries]
    
    return all_queries


# ============================================================================
# Environment Setup
# ============================================================================

def setup_environments(config: SimpleNamespace) -> Tuple:
    """
    Setup both tensor and compiled environments from the same data.
    
    Returns:
        (tensor_env, compiled_env, im, debug_helper, dh)
    """
    device = torch.device(config.device)
    
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file=config.train_file,
        valid_file=config.valid_file,
        test_file=config.test_file,
        rules_file=config.rules_file,
        facts_file=config.facts_file,
        train_depth=config.train_depth,
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_runtime_vars,
        padding_atoms=config.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
        rules=dh.rules,
    )
    dh.materialize_indices(im=im, device=device)
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    # Create base unification engine for tensor environment only
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=False,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx if config.end_proof_action else None,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.max_derived_per_state,
        sort_states=False,
    )
    base_engine.index_manager = im
    
    # Create vectorized engine for compiled env (with parity_mode=True)
    vec_engine = UnificationEngineVectorized.from_index_manager(
        im,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=config.padding_atoms,
        parity_mode=True,  # Enable exact matching
        max_derived_per_state=config.max_derived_per_state,
        end_proof_action=config.end_proof_action,
    )
    
    return base_engine, vec_engine, im, stringifier_params, dh


def create_tensor_env(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    base_engine: UnificationEngine,
    im: IndexManager,
    config: SimpleNamespace,
) -> BatchedEnv:
    """Create a tensor BatchedEnv for the given queries."""
    device = torch.device(config.device)
    batch_size = len(queries)
    
    # Convert queries to tensor format
    query_tensors = []
    for split, (p, h, t) in queries:
        query_atom = im.atom_to_tensor(p, h, t)
        query_padded = torch.full(
            (config.padding_atoms, 3), im.padding_idx,
            dtype=torch.long, device=device
        )
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)
    
    return BatchedEnv(
        batch_size=batch_size,
        queries=queries_tensor,
        labels=torch.ones(batch_size, dtype=torch.long, device=device),
        query_depths=torch.ones(batch_size, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=config.max_depth,
        memory_pruning=config.memory_pruning,
        use_exact_memory=config.use_exact_memory,
        skip_unary_actions=config.skip_unary_actions,  # Must be False
        end_proof_action=config.end_proof_action,
        reward_type=config.reward_type,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=config.prover_verbose,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + config.max_total_runtime_vars,
        sample_deterministic_per_env=False,
        stringifier_params=getattr(base_engine, 'stringifier_params', None),
    )


def create_compiled_env(
    vec_engine: UnificationEngineVectorized,
    im: IndexManager,
    config: SimpleNamespace,
    batch_size: int,
) -> EnvVec:
    """Create a compiled EvalOnlyEnvOptimized environment."""
    device = torch.device(config.device)
    
    return EnvVec(
        vec_engine=vec_engine,
        batch_size=batch_size,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
        memory_pruning=config.memory_pruning,
    )


# ============================================================================
# Query Running
# ============================================================================

def safe_item(x):
    """Safely extract scalar from numpy/torch/python scalar."""
    if isinstance(x, (np.ndarray, np.generic)):
        return x.item() if x.size == 1 else x
    elif isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x
    return x


def run_batch_traces(
    batch_queries: List[Tuple[str, Tuple[str, str, str]]],
    tensor_env: BatchedEnv,
    compiled_env: EnvVec,
    im: IndexManager,
    stringifier_params: Dict,
    config: SimpleNamespace,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """
    Run a batch of queries through both environments and collect step traces.
    
    Both environments are reused - just reset with new queries each batch.
    
    Args:
        batch_queries: List of (split, (pred, head, tail)) tuples
        tensor_env: Pre-created tensor environment (will be reset)
        compiled_env: Pre-created compiled environment (will be reinitialized)
        
    Returns:
        (all_tensor_traces, all_compiled_traces) - each is a list of traces per query
    """
    device = torch.device(config.device)
    pad = im.padding_idx
    batch_size = len(batch_queries)
    
    # Convert queries to tensor format
    query_tensors = []
    query_atoms = []
    for split, (p, h, t) in batch_queries:
        query_atom = im.atom_to_tensor(p, h, t)
        query_atoms.append(query_atom)
        query_padded = torch.full(
            (config.padding_atoms, 3), pad,
            dtype=torch.long, device=device
        )
        query_padded[0] = query_atom
        query_tensors.append(query_padded)
    
    queries_tensor = torch.stack(query_tensors, dim=0)  # [B_actual, A, 3]
    query_atoms_tensor = torch.stack(query_atoms, dim=0)  # [B_actual, 3]
    
    # Pad if necessary to match environment batch size
    env_batch_size = tensor_env.batch_size[0]
    padding_needed = env_batch_size - batch_size
    
    if padding_needed > 0:
        # Pad queries tensor
        padding_queries = queries_tensor[0:1].repeat(padding_needed, 1, 1)
        queries_tensor = torch.cat([queries_tensor, padding_queries], dim=0)
        
        # Pad atoms tensor
        padding_atoms_t = query_atoms_tensor[0:1].repeat(padding_needed, 1)
        query_atoms_tensor = torch.cat([query_atoms_tensor, padding_atoms_t], dim=0)
        
        effective_batch_size = env_batch_size
    else:
        effective_batch_size = batch_size
    
    # Reset tensor env with new queries (reuse the environment)
    tensor_env.set_eval_dataset(
        queries=queries_tensor,
        labels=torch.ones(effective_batch_size, dtype=torch.long, device=device),
        query_depths=torch.ones(effective_batch_size, dtype=torch.long, device=device),
        per_slot_lengths=torch.ones(effective_batch_size, dtype=torch.long, device=device),
    )
    tensor_obs = tensor_env.reset()
    if 'next' in tensor_obs.keys():
        tensor_obs = tensor_obs['next']
    
    # Reinitialize compiled env state with new queries (state is immutable)
    compiled_state = compiled_env._reset_from_queries(query_atoms_tensor)
    
    # Initialize trace storage: one list per query
    # Initialize trace storage: one list per query
    all_tensor_traces = [[] for _ in range(effective_batch_size)]
    all_compiled_traces = [[] for _ in range(effective_batch_size)]
    
    # Track which queries are done
    # Track which queries are done
    tensor_done = torch.zeros(effective_batch_size, dtype=torch.bool, device=device)
    compiled_done = torch.zeros(effective_batch_size, dtype=torch.bool, device=device)
    
    for step in range(config.max_depth):
        # ===== Process tensor env for all queries =====
        tensor_mask = tensor_obs['action_mask']  # [B, S]
        tensor_n_actions = tensor_mask.sum(dim=1).int()  # [B]
        
        for i in range(effective_batch_size):
            if tensor_done[i]:
                continue
            
            n_actions = int(tensor_n_actions[i])
            state_str = tensor_env.state_to_str(tensor_env.current_queries[i])
            
            if n_actions == 0:
                all_tensor_traces[i].append({
                    'step': step,
                    'state': state_str,
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                })
                tensor_done[i] = True
            else:
                derived_strs = sorted([
                    tensor_env.state_to_str(tensor_env.derived_states_batch[i, a])
                    for a in range(n_actions)
                ])
                all_tensor_traces[i].append({
                    'step': step,
                    'state': state_str,
                    'derived_states': derived_strs,
                    'num_actions': n_actions,
                    'action': 0,
                    'done': False,
                })
        
        # ===== Process compiled env for all queries =====
        compiled_counts = compiled_state['derived_counts']  # [B]
        
        for i in range(effective_batch_size):
            if compiled_done[i]:
                continue
            
            n_actions = int(compiled_counts[i])
            state_str = utils_funcs.state_to_str(compiled_state['current_states'][i], **stringifier_params)
            
            if n_actions == 0:
                all_compiled_traces[i].append({
                    'step': step,
                    'state': state_str,
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True,
                })
                compiled_done[i] = True
            else:
                derived_strs = sorted([
                    utils_funcs.state_to_str(compiled_state['derived_states'][i, k], **stringifier_params)
                    for k in range(n_actions)
                ])
                all_compiled_traces[i].append({
                    'step': step,
                    'state': state_str,
                    'derived_states': derived_strs,
                    'num_actions': n_actions,
                    'action': 0,
                    'done': False,
                })
        
        # Check if all done
        if tensor_done.all() and compiled_done.all():
            break
        
        # Take step for tensor env (action=0 for all)
        if not tensor_done.all():
            actions_tensor = torch.zeros(effective_batch_size, dtype=torch.long, device=device)
            action_td = TensorDict({'action': actions_tensor}, batch_size=[effective_batch_size])
            result_td = tensor_env.step(action_td)
            if 'next' in result_td.keys():
                tensor_obs = result_td['next']
            else:
                tensor_obs = result_td
            # Update done flags
            tensor_done = tensor_done | tensor_obs['done'].squeeze(-1)
        
        # Take step for compiled env (action=0 for all)
        if not compiled_done.all():
            actions_compiled = torch.zeros(effective_batch_size, dtype=torch.long, device=device)
            _, compiled_state = compiled_env.step(compiled_state, actions_compiled)
            # Update done flags
            compiled_done = compiled_done | compiled_state['done']
    
    return all_tensor_traces[:batch_size], all_compiled_traces[:batch_size]


# ============================================================================
# Trace Comparison
# ============================================================================

# Note: canonicalize_state is NOT used because with parity_mode=True in engines,
# variable names match exactly. This matches test_unification_compiled_parity behavior.


def compare_trace_step(step_tensor: Dict, step_compiled: Dict, step_idx: int) -> Tuple[bool, str]:
    """Compare a single step. Returns (match, error_message).
    
    With parity_mode=True in the engines, variable names should match exactly.
    No canonicalization needed (matching test_unification_compiled_parity behavior).
    """
    # Compare state strings directly (no canonicalization needed with parity_mode)
    tensor_state = step_tensor.get('state', '')
    compiled_state = step_compiled.get('state', '')
    
    if tensor_state != compiled_state:
        return False, f"Step {step_idx}: state mismatch:\n  Tensor:   {tensor_state}\n  Compiled: {compiled_state}"
    
    # Compare number of actions
    if step_tensor.get('num_actions') != step_compiled.get('num_actions'):
        return False, f"Step {step_idx}: num_actions mismatch: {step_tensor.get('num_actions')} vs {step_compiled.get('num_actions')}"
    
    # Compare derived states directly (no canonicalization needed with parity_mode)
    tensor_derived = set(step_tensor.get('derived_states', []))
    compiled_derived = set(step_compiled.get('derived_states', []))
    
    if tensor_derived != compiled_derived:
        only_tensor = tensor_derived - compiled_derived
        only_compiled = compiled_derived - tensor_derived
        return False, f"Step {step_idx}: derived_states mismatch:\n  Only tensor: {list(only_tensor)[:2]}\n  Only compiled:  {list(only_compiled)[:2]}"
    
    # Compare done flag
    if step_tensor.get('done') != step_compiled.get('done'):
        return False, f"Step {step_idx}: done mismatch: {step_tensor.get('done')} vs {step_compiled.get('done')}"
    
    return True, ""


def compare_full_traces(trace_tensor: List[Dict], trace_compiled: List[Dict]) -> Tuple[bool, str]:
    """Compare full traces step-by-step. Returns (match, error_message)."""
    if len(trace_tensor) != len(trace_compiled):
        return False, f"Trace length mismatch: {len(trace_tensor)} vs {len(trace_compiled)}"
    
    for i, (step_t, step_c) in enumerate(zip(trace_tensor, trace_compiled)):
        match, error = compare_trace_step(step_t, step_c, i)
        if not match:
            return False, error
    
    return True, ""


def compare_all_traces(
    all_tensor_traces: List[List[Dict]],
    all_compiled_traces: List[List[Dict]],
    queries: List[Tuple[str, Tuple[str, str, str]]]
) -> Tuple[int, List[Tuple[int, str, str]]]:
    """Compare all query traces. Returns (num_matches, list of mismatches)."""
    mismatches = []
    matches = 0
    
    for i, (tensor_trace, compiled_trace, query) in enumerate(zip(all_tensor_traces, all_compiled_traces, queries)):
        split, (pred, head, tail) = query
        query_str = f"{pred}({head}, {tail}) [{split}]"
        
        match, error = compare_full_traces(tensor_trace, compiled_trace)
        if match:
            matches += 1
        else:
            mismatches.append((i, query_str, error))
    
    return matches, mismatches


# ============================================================================
# Main Test Function
# ============================================================================

def run_parity_test(
    dataset: str,
    n_queries: int,
    max_depth: int = 20,
    seed: int = 42,
    compile_mode: bool = False,
    verbose: bool = False,
    batch_size: int = 100,
) -> Tuple[bool, Dict]:
    """Run parity test comparing tensor and compiled environments.
    
    Args:
        dataset: Dataset name
        n_queries: Number of queries to test
        max_depth: Maximum depth per query
        seed: Random seed
        compile_mode: Whether to use torch.compile
        verbose: Print detailed output
        batch_size: Batch size for processing queries
    """
    
    config = create_default_config()
    config.dataset = dataset
    config.n_queries = n_queries
    config.max_depth = max_depth
    config.seed = seed
    config.verbose = verbose
    config.skip_unary_actions = False  # Critical for parity
    config.batch_size = batch_size
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*80}")
    print(f"ENVIRONMENT COMPILED PARITY TEST")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Queries: {n_queries}")
    print(f"Max depth: {max_depth}")
    print(f"skip_unary_actions: {config.skip_unary_actions}")
    print(f"Compile mode: {compile_mode}")
    print(f"{'='*80}\n")
    
    # Setup both environments
    base_engine, vec_engine, im, stringifier_params, dh = setup_environments(config)
    
    # Prepare queries
    queries = prepare_queries(config)
    print(f"Testing {len(queries)} queries in batches of {config.batch_size}...\n")
    
    # Create environments ONCE (will be reused for all batches)
    # Use first batch_size queries as dummy queries to initialize tensor_env
    first_batch = queries[:config.batch_size]
    # Pad if needed (last batch might be smaller)
    while len(first_batch) < config.batch_size:
        first_batch.append(first_batch[0])  # Duplicate first query as padding
    
    tensor_env = create_tensor_env(first_batch, base_engine, im, config)
    compiled_env = create_compiled_env(vec_engine, im, config, config.batch_size)
    
    # Run queries in batches
    all_tensor_traces = []
    all_compiled_traces = []
    
    num_batches = (len(queries) + config.batch_size - 1) // config.batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * config.batch_size
        end_idx = min(start_idx + config.batch_size, len(queries))
        batch_queries = queries[start_idx:end_idx]
        
        # Run batch through both environments (envs are reused, just reset)
        batch_tensor_traces, batch_compiled_traces = run_batch_traces(
            batch_queries, tensor_env, compiled_env, im, stringifier_params, config
        )
        
        all_tensor_traces.extend(batch_tensor_traces)
        all_compiled_traces.extend(batch_compiled_traces)
        
        print(f"  Processed {end_idx}/{len(queries)} queries...")
    
    # Compare all traces
    matches, mismatches = compare_all_traces(all_tensor_traces, all_compiled_traces, queries)
    
    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total queries:  {len(queries)}")
    print(f"Matches:        {matches}")
    print(f"Mismatches:     {len(mismatches)}")
    print(f"{'='*80}")
    
    if mismatches:
        print(f"\nFirst {min(5, len(mismatches))} mismatches:")
        for idx, query_str, error in mismatches[:5]:
            print(f"\n  Query {idx}: {query_str}")
            print(f"    {error}")
    
    all_passed = len(mismatches) == 0
    
    if all_passed:
        print(f"\n✓ ALL {len(queries)} QUERIES MATCH")
    else:
        print(f"\n✗ {len(mismatches)} QUERIES HAVE MISMATCHES")
    
    return all_passed, {
        'total': len(queries),
        'matches': matches,
        'mismatches': mismatches,
    }


# ============================================================================
# Pytest Fixtures and Tests
# ============================================================================

@pytest.fixture(scope="module")
def base_config():
    return create_default_config()


class TestEnvCompiledParity:
    """Test that compiled environment produces same results as tensor environment."""
    
    @pytest.mark.parametrize("dataset,n_queries,batch_size", [
        ("countries_s3", 50, 100),
        ("countries_s3", 111, 100),  # All queries
        ("family", 50, 100),
        ("family", 200, 100),
    ])
    def test_env_traces_match(self, dataset: str, n_queries: int, batch_size: int, base_config):
        """Test that tensor and compiled environments produce identical traces.
        
        Args:
            dataset: Dataset name
            n_queries: Number of queries to test
            batch_size: Batch size for processing queries
        """
        passed, results = run_parity_test(
            dataset=dataset,
            n_queries=n_queries,
            max_depth=20,
            seed=base_config.seed,
            compile_mode=False,
            batch_size=batch_size,
        )
        
        if not passed:
            pytest.fail(f"Found {len(results['mismatches'])} mismatches out of {results['total']} queries")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test environment compiled parity')
    parser.add_argument('--dataset', type=str, default='countries_s3')
    parser.add_argument('--n-queries', type=int, default=111)
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing queries (default: 100)')
    parser.add_argument('--max-depth', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    passed, results = run_parity_test(
        dataset=args.dataset,
        n_queries=args.n_queries,
        max_depth=args.max_depth,
        seed=args.seed,
        compile_mode=args.compile,
        verbose=args.verbose,
        batch_size=args.batch_size,
    )
    
    sys.exit(0 if passed else 1)
