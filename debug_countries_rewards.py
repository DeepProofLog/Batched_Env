"""
Debug script to analyze why rewards are 0 for countries_s3.
Tests individual queries step by step with detailed logging.
"""
import sys
import os
import torch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from data_handler import DataHandler
from index_manager import IndexManager
from sampler import Sampler, SamplerConfig
from embeddings import get_embedder
from env import BatchedEnv
from unification_engine import UnificationEngine
from atom_stringifier import AtomStringifier


def debug_single_query(env, query_idx, idx_manager, dh, max_steps=20):
    """
    Debug a single query step by step.
    
    Args:
        env: The environment
        query_idx: Index of query to test
        idx_manager: IndexManager for debugging
        dh: DataHandler for debugging
        max_steps: Maximum steps to run
    """
    print("\n" + "="*80)
    print(f"DEBUGGING QUERY {query_idx}")
    print("="*80)
    
    # Reset to this specific query
    env.mode = 'eval'  # Use eval mode to get specific query
    
    # Manually set the query
    query = env.all_queries[query_idx]
    label = env.all_labels[query_idx]
    depth = env.all_depths[query_idx]
    
    print(f"\nQuery: {env.atom_stringifier.atom_to_str(query)}")
    print(f"Label: {label} (1=positive, 0=negative)")
    print(f"Depth: {depth}")
    
    # Set all environments to this query
    # The query should be [1, 3] (single atom)
    # But current_queries is [batch_size, padding_atoms, 3]
    # We need to pad it properly
    padded_query = env._pad_state(query.unsqueeze(0))  # [padding_atoms, 3]
    env.current_queries[0] = padded_query
    env.original_queries[0] = query  # [3]
    env.current_labels[0] = label
    env.current_depths[0] = 0
    env.next_var_indices[0] = env.runtime_var_start_index
    
    print(f"\nDebug query shapes:")
    print(f"  Original query shape: {query.shape}")
    print(f"  Padded query shape: {padded_query.shape}")
    print(f"  Padded query:\n{padded_query}")
    print(f"  Original queries: {env.original_queries[0]}")
    
    print(f"\nDebug query indices:")
    print(f"  Query tensor: {env.original_queries[0]}")
    print(f"  Query string: {env.atom_stringifier.atom_to_str(env.original_queries[0])}")
    
    # Check if this query exists in facts
    facts = env.unification_engine.facts_idx
    matching_facts = (facts == env.original_queries[0].unsqueeze(0)).all(dim=1)
    print(f"  Number of matching facts: {matching_facts.sum().item()}")
    if matching_facts.any():
        idx = matching_facts.nonzero()[0].item()
        print(f"  First matching fact at index {idx}: {env.atom_stringifier.atom_to_str(facts[idx])}")
    
    # Clear memories
    for i in range(env.batch_size_int):
        env.memories[i].clear()
    
    print(f"\nDebug UnificationEngine:")
    print(f"  Facts tensor shape: {env.unification_engine.facts_idx.shape}")
    print(f"  Rules tensor shape: {env.unification_engine.rules_idx.shape}")
    print(f"  Number of facts: {env.unification_engine.facts_idx.shape[0]}")
    print(f"  Number of rules: {env.unification_engine.rules_idx.shape[0]}")
    print(f"  True pred idx: {env.unification_engine.true_pred_idx}")
    print(f"  False pred idx: {env.unification_engine.false_pred_idx}")
    print(f"  exclude_query_from_facts: {env.exclude_query_from_facts}")
    
    # Print a few example facts
    print(f"\nFirst 5 facts:")
    for i in range(min(5, env.unification_engine.facts_idx.shape[0])):
        fact = env.unification_engine.facts_idx[i]
        print(f"  {i}: {env.atom_stringifier.atom_to_str(fact)}")
    
    # Print a few example rules
    print(f"\nFirst 3 rules (if rules_idx has proper dimensions):")
    if env.unification_engine.rules_idx.numel() > 0 and env.unification_engine.rules_idx.shape[1] > 0:
        for i in range(min(3, env.unification_engine.rules_idx.shape[0])):
            rule_body = env.unification_engine.rules_idx[i]
            rule_len = env.unification_engine.rule_lens[i].item()
            rule_head = env.unification_engine.rules_heads_idx[i]
            body = rule_body[:rule_len]
            print(f"  {i}: {env.atom_stringifier.atom_to_str(rule_head)} :- {env.atom_stringifier.state_to_str(body)}")
    else:
        print(f"  ERROR: Rules tensor has incorrect shape: {env.unification_engine.rules_idx.shape}")
        print(f"  This means rules were not loaded correctly!")
        print(f"  Rule lens: {env.unification_engine.rule_lens}")
    
    # Check the IndexManager directly
    print(f"\nChecking IndexManager:")
    print(f"  IndexManager rules_idx shape: {idx_manager.rules_idx.shape}")
    print(f"  IndexManager rule_lens: {idx_manager.rule_lens}")
    print(f"\nChecking DataHandler:")
    print(f"  Number of rules in dh.rules: {len(dh.rules)}")
    if len(dh.rules) > 0:
        print(f"  First rule: {dh.rules[0]}")
        print(f"  First rule head: {dh.rules[0].head}")
        print(f"  First rule body: {dh.rules[0].body}")
    
    # Get initial derived states
    print("\n" + "-"*80)
    print("STEP 0: Initial State")
    print("-"*80)
    print(f"Current State: {env.atom_stringifier.state_to_str(env.current_queries[0])}")
    
    batch_derived, batch_var_indices = env.unification_engine.get_derived_states(
        current_states=env.current_queries[0:1],
        next_var_indices=env.next_var_indices[0:1],
        excluded_queries=env.original_queries[0:1],
        labels=env.current_labels[0:1],
        verbose=2,  # Higher verbosity
    )
    
    print(f"\nRaw number of derived states BEFORE processing: {len(batch_derived[0])}")
    for i, ds in enumerate(batch_derived[0][:15]):  # Show first 15
        print(f"  {i}: {env.atom_stringifier.state_to_str(ds)}")
    if len(batch_derived[0]) > 15:
        print(f"  ... and {len(batch_derived[0]) - 15} more")
    
    # Process derived states
    all_derived = [batch_derived[0]]
    env.next_var_indices[0] = batch_var_indices[0]
    
    for i in range(env.batch_size_int):
        if all_derived[i] is None or len(all_derived[i]) == 0:
            all_derived[i] = []
        else:
            all_derived[i] = env._process_derived_states_for_env(i, all_derived[i], apply_skip_unary=True)
    
    # Create batched tensor
    batched_derived = torch.full(
        (env.batch_size_int, env.padding_states, env.padding_atoms, env.max_arity + 1),
        env.padding_idx,
        dtype=torch.long,
        device=env._device_internal
    )
    
    counts = torch.zeros(env.batch_size_int, dtype=torch.long, device=env._device_internal)
    
    for i, derived_states in enumerate(all_derived):
        count = min(len(derived_states), env.padding_states)
        counts[i] = count
        for j, state in enumerate(derived_states[:env.padding_states]):
            batched_derived[i, j] = env._pad_state(state)
    
    env.derived_states_batch = batched_derived
    env.derived_states_counts = counts
    
    print(f"\nProcessed derived states: {env.derived_states_counts[0].item()}")
    
    # Check if already done
    rewards, dones = env._get_done_reward_vec()
    print(f"\nInitial reward: {rewards[0, 0].item()}")
    print(f"Initial done: {dones[0, 0].item()}")
    
    if dones[0, 0].item():
        print("\nQuery completed at initialization!")
        return rewards[0, 0].item(), 0
    
    # Run episode
    total_reward = 0.0
    step = 0
    
    for step in range(max_steps):
        print("\n" + "-"*80)
        print(f"STEP {step + 1}")
        print("-"*80)
        
        # Take random action
        valid_actions = env.derived_states_counts[0].item()
        if valid_actions == 0:
            print("No valid actions available!")
            break
        
        action = torch.randint(0, valid_actions, (1,), device=env._device_internal)
        print(f"Taking random action: {action.item()} (out of {valid_actions} valid actions)")
        
        # Get the selected state
        selected_state = env.derived_states_batch[0, action.item()]
        print(f"Selected state: {env.atom_stringifier.state_to_str(selected_state)}")
        
        # Update current state
        env.current_queries[0] = selected_state
        env.current_depths[0] += 1
        
        print(f"Current depth: {env.current_depths[0].item()}/{env.max_depth}")
        
        # Get new derived states
        batch_derived, batch_var_indices = env.unification_engine.get_derived_states(
            current_states=env.current_queries[0:1],
            next_var_indices=env.next_var_indices[0:1],
            excluded_queries=env.original_queries[0:1],
            labels=env.current_labels[0:1],
            verbose=1,
        )
        
        print(f"\nNumber of derived states: {len(batch_derived[0])}")
        for i, ds in enumerate(batch_derived[0][:5]):  # Show first 5
            print(f"  {i}: {env.atom_stringifier.state_to_str(ds)}")
        if len(batch_derived[0]) > 5:
            print(f"  ... and {len(batch_derived[0]) - 5} more")
        
        # Process derived states
        all_derived = [batch_derived[0]]
        env.next_var_indices[0] = batch_var_indices[0]
        
        for i in range(env.batch_size_int):
            if all_derived[i] is None or len(all_derived[i]) == 0:
                all_derived[i] = []
            else:
                all_derived[i] = env._process_derived_states_for_env(i, all_derived[i], apply_skip_unary=True)
        
        # Create batched tensor
        batched_derived = torch.full(
            (env.batch_size_int, env.padding_states, env.padding_atoms, env.max_arity + 1),
            env.padding_idx,
            dtype=torch.long,
            device=env._device_internal
        )
        
        counts = torch.zeros(env.batch_size_int, dtype=torch.long, device=env._device_internal)
        
        for i, derived_states in enumerate(all_derived):
            count = min(len(derived_states), env.padding_states)
            counts[i] = count
            for j, state in enumerate(derived_states[:env.padding_states]):
                batched_derived[i, j] = env._pad_state(state)
        
        env.derived_states_batch = batched_derived
        env.derived_states_counts = counts
        
        print(f"\nProcessed derived states: {env.derived_states_counts[0].item()}")
        
        # Check rewards and dones
        rewards, dones = env._get_done_reward_vec()
        step_reward = rewards[0, 0].item()
        is_done = dones[0, 0].item()
        
        print(f"\nStep reward: {step_reward}")
        print(f"Done: {is_done}")
        
        total_reward += step_reward
        
        if is_done:
            print(f"\nEpisode completed at step {step + 1}!")
            print(f"Total reward: {total_reward}")
            break
    
    if not is_done:
        print(f"\nEpisode reached max steps ({max_steps})")
        print(f"Total reward: {total_reward}")
    
    return total_reward, step + 1


def test_multiple_queries(num_queries=10):
    """Test multiple queries and collect statistics."""
    print("\n" + "="*80)
    print("TESTING MULTIPLE QUERIES WITH RANDOM ACTIONS")
    print("="*80)
    
    # Configuration
    args = SimpleNamespace(
        dataset_name="countries_s3",
        max_depth=20,
        batch_size=1,  # Single environment for detailed debugging
        data_path="data",
        janus_file=None,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        n_train_queries=None,
        n_eval_queries=None,
        n_test_queries=None,
        corruption_mode=True,
        corruption_scheme=['tail'],
        train_neg_ratio=0,
        max_total_vars=1000,
        padding_atoms=6,
        padding_states=20,
        atom_embedder='transe',
        state_embedder='sum',
        constant_embedding_size=256,
        predicate_embedding_size=256,
        atom_embedding_size=256,
        learn_embeddings=True,
        variable_no=100,
        seed_run_i=42,
        lr=3e-4,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        engine='python_tensor',
        endt_action=False,
        endf_action=True,
        skip_unary_actions=True,
        memory_pruning=True,
        reward_type=1,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading dataset...")
    dh = DataHandler(
        dataset_name=args.dataset_name,
        base_path=args.data_path,
        janus_file=args.janus_file,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        rules_file=args.rules_file,
        facts_file=args.facts_file,
        n_train_queries=args.n_train_queries,
        n_eval_queries=args.n_eval_queries,
        n_test_queries=args.n_test_queries,
    )
    
    # Build IndexManager
    print("\nBuilding index manager...")
    idx_manager = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=args.max_total_vars,
        padding_atoms=args.padding_atoms,
        max_arity=dh.max_arity,
        device=device,
    )
    
    # Set facts and rules using IndexManager's direct methods
    print(f"  Setting facts and rules in IndexManager...")
    facts_tensor = idx_manager.state_to_tensor([(f.predicate, 
                                        f.args[0] if len(f.args) > 0 else '', 
                                        f.args[1] if len(f.args) > 1 else '') 
                                        for f in dh.facts])
    idx_manager.set_facts(facts_tensor)
    
    max_rule_atoms = max([len(r.body) for r in dh.rules]) if dh.rules else 8
    rules_as_tuples = [
        (
            (r.head.predicate, r.head.args[0] if len(r.head.args) > 0 else '', r.head.args[1] if len(r.head.args) > 1 else ''),
            [(t.predicate, t.args[0] if len(t.args) > 0 else '', t.args[1] if len(t.args) > 1 else '') for t in r.body]
        )
        for r in dh.rules
    ]
    rules_tensor, rule_lens = idx_manager.rules_to_tensor(rules_as_tuples, max_rule_atoms=max_rule_atoms)
    idx_manager.set_rules(rules_tensor, rule_lens)
    
    # Create sampler
    print("\nCreating negative sampler...")
    # Prepare all known triples for filtering
    all_triples = []
    for split_queries in [dh.train_queries, dh.valid_queries, dh.test_queries]:
        for q in split_queries:
            all_triples.append([
                idx_manager.predicate_str2idx[q.predicate],
                idx_manager.constant_str2idx[q.args[0]] if len(q.args) > 0 else 0,
                idx_manager.constant_str2idx[q.args[1]] if len(q.args) > 1 else 0
            ])
    all_triples_tensor = torch.tensor(all_triples, dtype=torch.long, device='cpu')
    
    # Determine default_mode from corruption_scheme
    if args.corruption_scheme is not None:
        if 'head' in args.corruption_scheme and 'tail' in args.corruption_scheme:
            default_mode = 'both'
        elif 'head' in args.corruption_scheme:
            default_mode = 'head'
        elif 'tail' in args.corruption_scheme:
            default_mode = 'tail'
        else:
            default_mode = 'both'
    else:
        default_mode = 'both'
    
    sampler = Sampler.from_data(
        all_known_triples_idx=all_triples_tensor,
        num_entities=idx_manager.constant_no,
        num_relations=idx_manager.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=args.seed_run_i,
    )
    
    # Create atom stringifier
    atom_stringifier = AtomStringifier.from_index_manager(idx_manager)
    
    # Create unification engine
    print("\nCreating unification engine...")
    engine = UnificationEngine.from_index_manager(idx_manager)
    
    # Create environment
    print("\nCreating environment...")
    train_queries_tensor = torch.stack([
        idx_manager.state_to_tensor([(q.predicate, 
                                    q.args[0] if len(q.args) > 0 else '', 
                                    q.args[1] if len(q.args) > 1 else '')])
        for q in dh.train_queries
    ]).squeeze(1)  # [N, 3]
    train_labels = [1 for _ in dh.train_queries]  # All positive initially
    train_depths = [d if d is not None else 0 for d in dh.train_depths]
    
    env = BatchedEnv(
        batch_size=args.batch_size,
        queries=train_queries_tensor,
        labels=train_labels,
        query_depths=train_depths,
        unification_engine=engine,
        sampler=sampler,
        max_arity=dh.max_arity,
        padding_idx=0,
        runtime_var_start_index=idx_manager.runtime_var_start_index,
        total_vocab_size=idx_manager.total_vocab_size,
        true_pred_idx=idx_manager.true_pred_idx,
        false_pred_idx=idx_manager.false_pred_idx,
        end_pred_idx=idx_manager.predicate_str2idx.get('End', None) if args.endf_action else None,
        atom_stringifier=atom_stringifier,
        mode='train',
        corruption_mode=args.corruption_mode,
        corruption_scheme=args.corruption_scheme,
        train_neg_ratio=args.train_neg_ratio,
        max_depth=args.max_depth,
        memory_pruning=args.memory_pruning,
        end_proof_action=args.endf_action,
        skip_unary_actions=args.skip_unary_actions,
        padding_atoms=args.padding_atoms,
        padding_states=args.padding_states,
        reward_type=args.reward_type,
        exclude_query_from_facts=False,
        verbose=1,
        prover_verbose=0,
        device=device,
    )
    
    # Test multiple queries
    rewards = []
    steps_taken = []
    
    for i in range(num_queries):
        reward, steps = debug_single_query(env, i, idx_manager, dh, max_steps=20)
        rewards.append(reward)
        steps_taken.append(steps)
        
        print(f"\nQuery {i}: Total Reward = {reward}, Steps = {steps}")
    
    # Print statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    print(f"Number of queries tested: {num_queries}")
    print(f"Average reward: {np.mean(rewards):.4f}")
    print(f"Min reward: {np.min(rewards):.4f}")
    print(f"Max reward: {np.max(rewards):.4f}")
    print(f"Std reward: {np.std(rewards):.4f}")
    print(f"Rewards distribution: {np.unique(rewards, return_counts=True)}")
    print(f"\nAverage steps: {np.mean(steps_taken):.2f}")
    print(f"Min steps: {np.min(steps_taken)}")
    print(f"Max steps: {np.max(steps_taken)}")
    
    # Check if average reward is in valid range
    avg_reward = np.mean(rewards)
    if avg_reward == 0:
        print("\n" + "!"*80)
        print("ERROR: Average reward is 0! This should not happen with random actions.")
        print("!"*80)
        return False
    elif 0 < avg_reward < 1:
        print("\n" + "="*80)
        print(f"SUCCESS: Average reward {avg_reward:.4f} is in valid range (0, 1)")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print(f"Average reward {avg_reward:.4f} is outside expected range (0, 1)")
        print("="*80)
        return False


if __name__ == "__main__":
    # Test a few queries in detail
    success = test_multiple_queries(num_queries=20)
    
    if not success:
        print("\nTest FAILED: Average rewards are not in expected range.")
        sys.exit(1)
    else:
        print("\nTest PASSED: Average rewards are in expected range.")
        sys.exit(0)
