"""
Test module for tensor-based unification engine.

Simple and modular testing for the tensor-based unification engine in eval mode.
"""
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_path)

import random
import torch
from typing import Tuple, Dict, List
from types import SimpleNamespace

from data_handler import DataHandler
from index_manager import IndexManager
from unification import UnificationEngine
from utils.debug_helper import DebugHelper


def get_default_tensor_engine_config() -> SimpleNamespace:
    return SimpleNamespace(
        padding_atoms=20,
        max_total_runtime_vars=1_000_000,
        max_derived_per_state=500,
        device='cpu'
    )


def setup_tensor_engine(
    dataset: str = "countries_s3",
    base_path: str = "./data/",
    batched: bool = False,
    config: SimpleNamespace = None
) -> Tuple:
    """
    Setup the tensor-based engine with dataset.
    
    Args:
        dataset: Dataset name
        base_path: Base path to data directory
        batched: If True, setup for batched operations
    
    Returns:
        (dh, im, engine, debug_helper, next_var_start)
    """
    cfg = config or get_default_tensor_engine_config()
    device_value = getattr(cfg, 'device', 'cpu')
    device = device_value if isinstance(device_value, torch.device) else torch.device(device_value)
    padding_atoms = getattr(cfg, 'padding_atoms', 20)
    max_total_runtime_vars = getattr(cfg, 'max_total_runtime_vars', 1_000_000)
    max_derived_per_state = getattr(cfg, 'max_derived_per_state', 500)

    dh_non = DataHandler(
        dataset_name=dataset,
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        train_depth=None,
    )

    im_non = IndexManager(
        constants=dh_non.constants,
        predicates=dh_non.predicates,
        max_total_runtime_vars=max_total_runtime_vars,
        padding_atoms=padding_atoms,
        max_arity=dh_non.max_arity,
        device=device,
        rules=dh_non.rules,
    )
    dh_non.materialize_indices(im=im_non, device=device)

    # Create stringifier params for engine initialization
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im_non.idx2predicate,
        'idx2constant': im_non.idx2constant,
        'idx2template_var': im_non.idx2template_var,
        'padding_idx': im_non.padding_idx,
        'n_constants': im_non.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(
        im_non, take_ownership=True, stringifier_params=stringifier_params,
        max_derived_per_state=max_derived_per_state,  # Set max derived states for eval mode
        sort_states=True
    )

    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im_non.idx2predicate,
        idx2constant=im_non.idx2constant,
        idx2template_var=im_non.idx2template_var,
        padding_idx=engine.padding_idx,
        n_constants=im_non.constant_no
    )

    # Compute max template variable index for initializing next_var during proof search
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start_for_proofs = max_template_var + 1

    return dh_non, im_non, engine, debug_helper, next_var_start_for_proofs


def test_tensor_engine_single_query(
    query_tuple: Tuple[str, str, str],
    engine_data: Tuple,
    split: str = 'train',
    deterministic: bool = True,
    max_depth: int = 10,
    max_derived_states: int = 200,
    verbose: bool = False,
    seed: int = 42
) -> Dict:
    """
    Test a single query using the tensor engine.
    
    Args:
        query_tuple: (predicate, head, tail)
        engine_data: Tuple from setup_tensor_engine()
        split: 'train', 'valid', or 'test'
        deterministic: If True, use canonical ordering; if False, random actions
        max_depth: Maximum proof depth
        max_derived_states: Maximum derived states per step
        verbose: Print detailed information
        seed: Random seed for reproducible random actions
        
    Returns:
        Dict with keys:
            - success: bool
            - steps: int
            - reward: float (1.0 if success, 0.0 otherwise)
            - trace: List[Dict] with state, derived_states, action at each step
    """
    p, h, t = query_tuple
    dh_non, im_non, engine, debug_helper, next_var_start = engine_data
    
    if verbose:
        print(f"\nQuery: {p}({h}, {t}) [split={split}, deterministic={deterministic}]")
    
    # Setup query
    query_tensor = im_non.atom_to_tensor(p, h, t)
    query_padded_tensor = query_tensor.unsqueeze(0).unsqueeze(0)
    padding = torch.zeros(1, 19, 3, dtype=torch.long, device='cpu')
    padding[:, :, 0] = engine.padding_idx
    query_padded = torch.cat([query_padded_tensor, padding], dim=1)
    
    # For training queries, exclude the query from facts
    excluded_query_tensor = query_padded if split == 'train' else None
    
    # Initialize next_var tracker
    next_var_tracker = torch.tensor([next_var_start], dtype=torch.long, device='cpu')
    
    # Proof functions
    def tensor_get_derived(state, next_var):
        if state.dim() == 2:
            state = state.unsqueeze(0)
        
        derived, derived_counts, updated_next_var = engine.get_derived_states(
            state, next_var,
            excluded_queries=excluded_query_tensor, verbose=0
        )
        num_derived = derived_counts[0].item()
        
        # NOTE: Do NOT filter out False() states here - match SB3 behavior!
        # The SB3 test engine does not filter False() states, it just continues
        # with them as regular actions. Filtering happens only in is_true checks.
        valid = []
        for i in range(num_derived):
            s = derived[0, i]
            non_padding = (s[:, 0] != engine.padding_idx).sum().item()
            if non_padding <= 100:
                valid.append(s.unsqueeze(0))
        return valid, updated_next_var
    
    def tensor_is_true(state):
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_true_state(state)
    
    def tensor_is_false(state):
        if state.dim() == 3:
            state = state.squeeze(0)
        return engine.is_false_state(state)
    
    def tensor_state_to_str(state):
        """Convert tensor state to string (preserves order, normalizes variables)."""
        s = state.squeeze(0) if state.dim() > 2 else state
        return debug_helper.state_to_str(s)
    
    # Run proof
    current_state = query_padded
    steps = 0
    trace = []
    rng = random.Random(seed)
    
    while steps < max_depth:
        # Check if already proved
        if tensor_is_true(current_state):
            trace.append({
                'step': steps,
                'state': tensor_state_to_str(current_state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True
            })
            if verbose:
                print(f"  ✓ Proved at step {steps}")
            return {
                'success': True,
                'steps': steps,
                'reward': 1.0,
                'trace': trace
            }
        
        # Get derived states
        tensor_derived, next_var_tracker = tensor_get_derived(current_state, next_var_tracker)
        
        if not tensor_derived:
            trace.append({
                'step': steps,
                'state': tensor_state_to_str(current_state),
                'derived_states': [],
                'num_actions': 0,
                'action': None,
                'done': True
            })
            if verbose:
                print(f"  ✗ Failed at step {steps} (no derived states)")
            return {
                'success': False,
                'steps': steps,
                'reward': 0.0,
                'trace': trace
            }
        
        # States are already sorted by the engine when sort_states=True
        # Choose action based on deterministic flag
        if deterministic:
            # Choose first state (already in canonical order)
            chosen_idx = 0
        else:
            # Choose random state
            chosen_idx = rng.randint(0, len(tensor_derived) - 1)
        
        chosen_state = tensor_derived[chosen_idx]
        
        trace.append({
            'step': steps,
            'state': tensor_state_to_str(current_state),
            'derived_states': [tensor_state_to_str(s) for s in tensor_derived],
            'num_actions': len(tensor_derived),
            'action': chosen_idx,
            'done': False
        })
        
        if verbose and steps < 3:
            print(f"  Step {steps}: {len(tensor_derived)} actions, chose {chosen_idx}")
        
        current_state = chosen_state
        steps += 1
    
    # Reached max depth
    success = tensor_is_true(current_state)
    trace.append({
        'step': steps,
        'state': tensor_state_to_str(current_state),
        'derived_states': [],
        'num_actions': 0,
        'action': None,
        'done': True
    })
    
    if verbose:
        print(f"  Max depth reached: success={success}")
    
    return {
        'success': success,
        'steps': steps,
        'reward': 1.0 if success else 0.0,
        'trace': trace
    }


def _run_tensor_engine_batch(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    engine_data: Tuple,
    config: SimpleNamespace,
    base_seed_offset: int = 0
) -> List[Dict]:
    """
    Run tensor engine on a single batch of queries.
    
    Args:
        queries: List of (split, (predicate, head, tail)) for this batch
        engine_data: Tuple from setup_tensor_engine()
        config: Configuration namespace
        base_seed_offset: Offset for random seeds to ensure consistency across batches
        
    Returns:
        List of result dicts for each query in the batch
    """
    dh_non, im_non, engine, debug_helper, next_var_start = engine_data
    
    deterministic = config.deterministic
    max_depth = config.max_depth
    seed = config.seed
    verbose = config.verbose
    
    B = len(queries)
    device = engine.device
    pad = engine.padding_idx
    
    # Start with initial padding but allow dynamic growth
    initial_max_atoms = 20
    
    # Initialize RNGs for non-deterministic action selection (one per query)
    rngs = [random.Random(seed + base_seed_offset + i) for i in range(B)]
    
    # Build initial states batch: [B, max_atoms, 3]
    current_states = torch.full((B, initial_max_atoms, 3), pad, dtype=torch.long, device=device)
    excluded_queries = torch.full((B, initial_max_atoms, 3), pad, dtype=torch.long, device=device)
    
    for i, (split, (p, h, t)) in enumerate(queries):
        query_tensor = im_non.atom_to_tensor(p, h, t)  # [3]
        current_states[i, 0] = query_tensor
        if split == 'train':
            excluded_queries[i, 0] = query_tensor
    
    # Initialize tracking
    next_var_tracker = torch.full((B,), next_var_start, dtype=torch.long, device=device)
    
    # Per-query tracking
    done = torch.zeros(B, dtype=torch.bool, device=device)
    success = torch.zeros(B, dtype=torch.bool, device=device)
    steps_taken = torch.zeros(B, dtype=torch.long, device=device)
    
    # Traces: list of dicts per query
    traces = [[] for _ in range(B)]
    
    def state_to_str(state):
        """Convert a single state tensor to string (preserves order, normalizes variables)."""
        s = state.squeeze(0) if state.dim() > 2 else state
        return debug_helper.state_to_str(s)
    
    def is_true(state):
        """Check if a single state is True."""
        s = state.squeeze(0) if state.dim() > 2 else state
        return engine.is_true_state(s)
    
    def is_false(state):
        """Check if a single state is False."""
        s = state.squeeze(0) if state.dim() > 2 else state
        return engine.is_false_state(s)
    
    # Main proof loop
    for step in range(max_depth + 1):
        # Check for queries that are already TRUE
        for i in range(B):
            if done[i]:
                continue
            if is_true(current_states[i]):
                done[i] = True
                success[i] = True
                traces[i].append({
                    'step': step,
                    'state': state_to_str(current_states[i]),
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True
                })
        
        # If all done, break early
        if done.all():
            break
        
        if step >= max_depth:
            # Max depth reached - mark remaining as done
            for i in range(B):
                if not done[i]:
                    done[i] = True
                    success[i] = is_true(current_states[i])
                    traces[i].append({
                        'step': step,
                        'state': state_to_str(current_states[i]),
                        'derived_states': [],
                        'num_actions': 0,
                        'action': None,
                        'done': True
                    })
            break
        
        # Get derived states for queries in this batch
        derived_states, derived_counts, updated_next_var = engine.get_derived_states(
            current_states, next_var_tracker,
            excluded_queries=excluded_queries, verbose=0
        )
        # derived_states: [B, K, M, 3], derived_counts: [B]
        
        next_var_tracker = updated_next_var
        
        # Get the output atom dimension from derived states
        M_out = derived_states.shape[2]
        
        # Resize current_states if derived states have more atoms
        current_max_atoms = current_states.shape[1]
        if M_out > current_max_atoms:
            new_current = torch.full((B, M_out, 3), pad, dtype=torch.long, device=device)
            new_current[:, :current_max_atoms, :] = current_states
            current_states = new_current
            
            # Also resize excluded_queries to match
            new_excluded = torch.full((B, M_out, 3), pad, dtype=torch.long, device=device)
            new_excluded[:, :current_max_atoms, :] = excluded_queries[:, :current_max_atoms, :]
            excluded_queries = new_excluded
        
        # Process each query's derived states
        for i in range(B):
            if done[i]:
                continue
            
            num_derived = derived_counts[i].item()
            
            # Collect valid derived states for this query
            valid_states = []
            for k in range(num_derived):
                s = derived_states[i, k]  # [M, 3]
                # Check if it's a padding state (all padding)
                non_padding = (s[:, 0] != pad).sum().item()
                if non_padding > 0 and non_padding <= 100:
                    valid_states.append(s)
            
            if not valid_states:
                # No valid derived states - mark as failed
                done[i] = True
                success[i] = False
                traces[i].append({
                    'step': step,
                    'state': state_to_str(current_states[i]),
                    'derived_states': [],
                    'num_actions': 0,
                    'action': None,
                    'done': True
                })
                continue
            
            # States are already sorted by the engine when sort_states=True
            # Choose action based on deterministic flag
            if deterministic:
                chosen_idx = 0
            else:
                chosen_idx = rngs[i].randint(0, len(valid_states) - 1)
            
            chosen_state = valid_states[chosen_idx]
            
            # Record trace
            traces[i].append({
                'step': step,
                'state': state_to_str(current_states[i]),
                'derived_states': [state_to_str(s) for s in valid_states],
                'num_actions': len(valid_states),
                'action': chosen_idx,
                'done': False
            })
            
            # Update state for this query
            M_chosen = chosen_state.shape[0]
            current_max_atoms = current_states.shape[1]
            
            # Reset the row to padding, then copy chosen_state
            current_states[i] = pad
            current_states[i, :M_chosen] = chosen_state
            
            steps_taken[i] += 1
    
    # Build results for this batch
    results = []
    for i in range(B):
        results.append({
            'success': success[i].item(),
            'steps': len(traces[i]) - 1 if traces[i] else 0,
            'reward': 1.0 if success[i].item() else 0.0,
            'trace': traces[i]
        })
    
    return results


def run_tensor_engine(
    queries: List[Tuple[str, Tuple[str, str, str]]],
    engine_data: Tuple,
    config: SimpleNamespace
) -> Dict:
    """
    Test multiple queries using the tensor engine in batches to avoid OOM.
    
    Queries are processed in batches (default 250) using the batched unification engine.
    Action selection (deterministic=first canonical, or random) is handled per query.
    
    Args:
        queries: List of (split, (predicate, head, tail))
        engine_data: Tuple from setup_tensor_engine()
        config: Configuration namespace with deterministic, max_depth, seed, verbose, etc.
            - batch_size: Number of queries to process at once (default: 250)
        
    Returns:
        Dict with keys:
            - total_queries: int
            - successful: int
            - avg_reward: float
            - avg_steps: float
            - traces: List of trace dicts
    """
    batch_size = getattr(config, 'batch_size', 250)
    total_queries = len(queries)
    
    all_results = []
    
    # Process queries in batches
    for batch_start in range(0, total_queries, batch_size):
        print(f"Processing queries {batch_start} to {min(batch_start + batch_size, total_queries)}...")
        batch_end = min(batch_start + batch_size, total_queries)
        batch_queries = queries[batch_start:batch_end]
        
        # Run this batch
        batch_results = _run_tensor_engine_batch(
            batch_queries, 
            engine_data, 
            config,
            base_seed_offset=batch_start
        )
        all_results.extend(batch_results)
        
        # Explicit cleanup to free memory between batches
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Aggregate statistics
    successful = sum(1 for r in all_results if r['success'])
    total_reward = sum(r['reward'] for r in all_results)
    total_steps = sum(r['steps'] for r in all_results)
    
    # Compute average actions (branching factor)
    total_actions = 0
    total_action_steps = 0
    for r in all_results:
        for step_data in r['trace']:
            if 'num_actions' in step_data:
                total_actions += step_data['num_actions']
                total_action_steps += 1
    
    avg_actions = total_actions / total_action_steps if total_action_steps > 0 else 0.0
    
    return {
        'total_queries': len(queries),
        'successful': successful,
        'avg_reward': total_reward / len(queries) if queries else 0.0,
        'avg_steps': total_steps / len(queries) if queries else 0.0,
        'avg_actions': avg_actions,
        'traces': all_results
    }