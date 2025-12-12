"""
Debug script to investigate MRR discrepancy between original and compiled environments.

This script traces step-by-step execution to find where the environments diverge.

Usage:
    python tests/debug_mrr_discrepancy.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from types import SimpleNamespace


def setup_components(device: torch.device, config: SimpleNamespace):
    """Initialize all components needed for evaluation."""
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine
    from unification_vectorized import UnificationEngineVectorized
    from embeddings import EmbedderLearnable as TensorEmbedder
    from model import ActorCriticPolicy as TensorPolicy
    from sampler import Sampler
    
    # Load data
    dh = DataHandler(
        dataset_name=config.dataset,
        base_path=config.data_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    # Index manager
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=config.max_total_vars,
        max_arity=dh.max_arity,
        padding_atoms=config.padding_atoms,
        device=device,
        rules=dh.rules,
    )
    
    # Materialize indices
    dh.materialize_indices(im=im, device=device)
    
    # Sampler
    default_mode = 'tail' if 'countries' in config.dataset else 'both'
    domain2idx, entity2domain = dh.get_sampler_domain_info()
    sampler = Sampler.from_data(
        all_known_triples_idx=dh.all_known_triples_idx,
        num_entities=im.constant_no,
        num_relations=im.predicate_no,
        device=device,
        default_mode=default_mode,
        seed=config.seed,
        domain2idx=domain2idx,
        entity2domain=entity2domain,
    )
    
    # Reseed
    torch.manual_seed(config.seed)
    
    # Base unification engine
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im.idx2predicate,
        'idx2constant': im.idx2constant,
        'idx2template_var': im.idx2template_var,
        'padding_idx': im.padding_idx,
        'n_constants': im.constant_no
    }
    
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=config.end_proof_action,
        max_derived_per_state=config.padding_states,
    )
    base_engine.index_manager = im
    
    # Create vectorized/compiled engine
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=config.max_fact_pairs,
        max_rule_pairs=config.max_rule_pairs,
    )
    
    # Convert queries
    def convert_queries_unpadded(queries):
        tensors = []
        for q in queries:
            atom = im.atom_to_tensor(q.predicate, q.args[0], q.args[1])
            tensors.append(atom)
        return torch.stack(tensors, dim=0)
    
    test_queries = convert_queries_unpadded(dh.test_queries)
    
    return {
        'base_engine': base_engine,
        'vec_engine': vec_engine,
        'sampler': sampler,
        'dh': dh,
        'im': im,
        'test_queries': test_queries,
    }


def compare_derived_states(base_states, base_counts, compiled_states, compiled_counts, im, env_idx=0, max_show=10):
    """Compare derived states from both engines."""
    print(f"\n{'='*60}")
    print(f"Environment {env_idx} - Comparing Derived States")
    print(f"{'='*60}")
    
    pad = im.padding_idx
    
    print(f"Base engine: {base_counts[env_idx].item()} derived states")
    print(f"Compiled engine: {compiled_counts[env_idx].item()} derived states")
    
    n_base = int(base_counts[env_idx].item())
    n_comp = int(compiled_counts[env_idx].item())
    
    print(f"\nBase derived states (showing first {min(n_base, max_show)}):")
    for i in range(min(n_base, max_show)):
        state = base_states[env_idx, i]
        atoms = []
        for j in range(state.shape[0]):
            atom = state[j].tolist()
            if atom[0] != pad:
                atoms.append(atom)
        print(f"  [{i}]: {atoms}")
    
    print(f"\nCompiled derived states (showing first {min(n_comp, max_show)}):")
    for i in range(min(n_comp, max_show)):
        state = compiled_states[env_idx, i]
        atoms = []
        for j in range(state.shape[0]):
            atom = state[j].tolist()
            if atom[0] != pad:
                atoms.append(atom)
        print(f"  [{i}]: {atoms}")
    
    # Check for matches
    matches = 0
    for i in range(min(n_base, n_comp)):
        if torch.equal(base_states[env_idx, i], compiled_states[env_idx, i]):
            matches += 1
    
    print(f"\nMatching states: {matches}/{min(n_base, n_comp)}")
    return matches == min(n_base, n_comp) and n_base == n_comp


def debug_step_by_step(components, config, device, n_queries=1, max_steps=3):
    """Run both environments step-by-step and compare."""
    from env_eval import EvalOnlyEnv
    from env_eval_compiled import EvalOnlyEnvCompiled
    from tensordict import TensorDict
    
    base_engine = components['base_engine']
    vec_engine = components['vec_engine']
    im = components['im']
    
    test_queries = components['test_queries'][:n_queries].to(device)
    
    print("\n" + "="*70)
    print("STEP-BY-STEP COMPARISON: Original vs Compiled Environment")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # Setup Original Environment
    # -------------------------------------------------------------------------
    orig_env = EvalOnlyEnv(
        unification_engine=base_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    orig_env.set_queries(test_queries)
    
    # -------------------------------------------------------------------------
    # Setup Compiled Environment
    # -------------------------------------------------------------------------
    comp_env = EvalOnlyEnvCompiled(
        vec_engine=vec_engine,
        batch_size=n_queries,
        padding_atoms=config.padding_atoms,
        padding_states=config.padding_states,
        max_depth=config.max_depth,
        end_proof_action=config.end_proof_action,
        runtime_var_start_index=im.constant_no + 1,
        device=device,
    )
    comp_env.set_queries(test_queries)
    
    print(f"\nQuery: {test_queries[0].tolist()}")
    pred_idx = test_queries[0, 0].item()
    arg0_idx = test_queries[0, 1].item()
    arg1_idx = test_queries[0, 2].item()
    print(f"  -> Predicate [{pred_idx}]: {im.idx2predicate[pred_idx] if pred_idx < len(im.idx2predicate) else '?'}")
    print(f"  -> Arg0 [{arg0_idx}]: {im.idx2constant[arg0_idx] if arg0_idx < len(im.idx2constant) else '?'}")
    print(f"  -> Arg1 [{arg1_idx}]: {im.idx2constant[arg1_idx] if arg1_idx < len(im.idx2constant) else '?'}")
    
    # -------------------------------------------------------------------------
    # STEP 0: Reset both environments
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("STEP 0: Reset / Initial State")
    print("-"*60)
    
    orig_obs = orig_env.reset_batch(n_envs=n_queries)
    comp_obs = comp_env.reset_batch(n_envs=n_queries)
    
    print(f"\nOriginal current_state: {orig_env._current_states[0, 0].tolist()}")
    print(f"Compiled current_state: {comp_env._current_states[0, 0].tolist()}")
    
    print(f"\nOriginal next_var_index: {orig_env._next_var_indices[0].item()}")
    print(f"Compiled next_var_index: {comp_env._next_var_indices[0].item()}")
    
    # Additional debug: Show fact lookup info
    print(f"\nDebug: Fact segment info for predicate 4 (first atom pred):")
    pred = 4
    if hasattr(vec_engine, 'fact_seg_starts') and pred < len(vec_engine.fact_seg_starts):
        start = vec_engine.fact_seg_starts[pred].item()
        length = vec_engine.fact_seg_lens[pred].item()
        print(f"  Compiled fact_seg: start={start}, len={length}")
        if length > 0:
            facts = vec_engine.facts_idx[start:start+min(length, 5)]
            print(f"  First few facts for pred {pred}: {facts.tolist()}")
            
            # Check if fact [4, 1101, 2706] exists
            search_fact = torch.tensor([[4, 1101, 2706]], device=device)
            found = base_engine.fact_index.contains(search_fact)
            print(f"  Fact [4, 1101, 2706] exists: {found.item()}")
            
            # Check fact [4, ?, 2706] for any ?
            all_facts = vec_engine.facts_idx[start:start+length]
            matching = (all_facts[:, 0] == 4) & (all_facts[:, 2] == 2706)
            if matching.any():
                matches = all_facts[matching]
                print(f"  Facts with pred=4 and arg1=2706: {matches.tolist()}")
                
                # Check position of [4, 1101, 2706] in the segment
                for i, fact in enumerate(all_facts):
                    if fact.tolist() == [4, 1101, 2706]:
                        actual_max = vec_engine.max_fact_pairs
                        print(f"  Position of [4, 1101, 2706] in segment: {i} (vec_engine.max_fact_pairs={actual_max})")
                        if i >= actual_max:
                            print(f"  WARNING: Position {i} >= max_fact_pairs {actual_max} - FACT WILL BE MISSED!")
                        else:
                            print(f"  OK: Position {i} < max_fact_pairs {actual_max}")
                        break
    
    # Show original engine's predicate range map
    if hasattr(base_engine, 'predicate_range_map') and base_engine.predicate_range_map is not None:
        prm = base_engine.predicate_range_map
        if pred < prm.shape[0]:
            start = prm[pred, 0].item()
            end = prm[pred, 1].item()
            print(f"  Original predicate_range_map[{pred}]: [{start}, {end})")
    
    match = compare_derived_states(
        orig_env._derived_states, orig_env._derived_counts,
        comp_env._derived_states, comp_env._derived_counts,
        im, env_idx=0
    )
    print(f"\nStep 0 derived states match: {match}")
    
    # -------------------------------------------------------------------------
    # STEP 1+: Take same actions and compare
    # -------------------------------------------------------------------------
    for step in range(1, max_steps + 1):
        print("\n" + "-"*60)
        print(f"STEP {step}")
        print("-"*60)
        
        # Get valid actions from original
        orig_valid = orig_obs['action_mask']
        n_valid_orig = orig_valid[0].sum().item()
        
        comp_valid = comp_obs.action_mask
        n_valid_comp = comp_valid[0].sum().item()
        
        print(f"Original valid actions: {n_valid_orig}")
        print(f"Compiled valid actions: {n_valid_comp}")
        
        if n_valid_orig == 0 or n_valid_comp == 0:
            print("No valid actions - environment terminated")
            break
        
        # Take action 0 (first valid action) in both
        actions = torch.zeros(n_queries, dtype=torch.long, device=device)
        
        print(f"Taking action: {actions[0].item()}")
        
        # Show which state we're selecting
        print(f"\nOriginal selected next state:")
        next_orig = orig_env._derived_states[0, actions[0].item()]
        atoms_orig = []
        for j in range(next_orig.shape[0]):
            atom = next_orig[j].tolist()
            if atom[0] != im.padding_idx:
                atoms_orig.append(atom)
        print(f"  {atoms_orig}")
        
        print(f"Compiled selected next state:")
        next_comp = comp_env._derived_states[0, actions[0].item()]
        atoms_comp = []
        for j in range(next_comp.shape[0]):
            atom = next_comp[j].tolist()
            if atom[0] != im.padding_idx:
                atoms_comp.append(atom)
        print(f"  {atoms_comp}")
        
        # Step both environments
        orig_obs, orig_rewards, orig_dones, orig_success = orig_env.step(actions)
        comp_result = comp_env.step_compiled(actions)
        comp_obs = comp_result.obs
        
        print(f"\nAfter step {step}:")
        # Show all atoms in current state
        orig_atoms = []
        for j in range(orig_env._current_states.shape[1]):
            atom = orig_env._current_states[0, j].tolist()
            if atom[0] != im.padding_idx:
                orig_atoms.append(atom)
        print(f"  Original current_state (full): {orig_atoms}")
        
        comp_atoms = []
        for j in range(comp_env._current_states.shape[1]):
            atom = comp_env._current_states[0, j].tolist()
            if atom[0] != im.padding_idx:
                comp_atoms.append(atom)
        print(f"  Compiled current_state (full): {comp_atoms}")
        
        print(f"  Original next_var_index: {orig_env._next_var_indices[0].item()}")
        print(f"  Compiled next_var_index: {comp_env._next_var_indices[0].item()}")
        
        print(f"  Original done: {orig_dones[0].item()}, success: {orig_success[0].item()}")
        print(f"  Compiled done: {comp_result.dones[0].item()}, success: {comp_result.success[0].item()}")
        
        if orig_dones[0].item() or comp_result.dones[0].item():
            print("\nOne or both environments terminated")
            if orig_dones[0].item() != comp_result.dones[0].item():
                print("WARNING: Done states differ!")
            break
        
        match = compare_derived_states(
            orig_env._derived_states, orig_env._derived_counts,
            comp_env._derived_states, comp_env._derived_counts,
            im, env_idx=0
        )
        print(f"\nStep {step} derived states match: {match}")
        
        if not match:
            print("\n*** DIVERGENCE DETECTED ***")
            break
    
    print("\n" + "="*70)
    print("DEBUG SESSION COMPLETE")
    print("="*70)


def main():
    # Configuration
    config = SimpleNamespace(
        dataset='family',  # Use family for more interesting proofs
        data_path='./data/',
        batch_size=1,
        padding_atoms=6,
        padding_states=120,
        max_depth=20,
        end_proof_action=True,
        max_total_vars=100,
        max_fact_pairs=None,  # Auto-compute from data
        max_rule_pairs=None,  # Auto-compute from data
        seed=42,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nSetting up components...")
    components = setup_components(device, config)
    
    print("\nRunning step-by-step debug...")
    debug_step_by_step(components, config, device, n_queries=1, max_steps=5)


if __name__ == '__main__':
    main()
