"""
Compare original vs vectorized unification at each step to see where they diverge.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from tensordict import TensorDict

def main():
    from data_handler import DataHandler
    from index_manager import IndexManager
    from unification import UnificationEngine, set_compile_mode
    from unification_vectorized import UnificationEngineVectorized
    from env import BatchedEnv
    from env_optimized import EvalEnvOptimized
    
    set_compile_mode(False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dh = DataHandler(
        dataset_name='family',
        base_path='./data/',
        train_file="train.txt",
        valid_file="valid.txt", 
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file="train.txt",
        corruption_mode='dynamic',
    )
    
    im = IndexManager(
        constants=dh.constants,
        predicates=dh.predicates,
        max_total_runtime_vars=100,
        max_arity=dh.max_arity,
        padding_atoms=6,
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
    
    base_engine = UnificationEngine.from_index_manager(
        im, take_ownership=True,
        stringifier_params=stringifier_params,
        end_pred_idx=im.end_pred_idx,
        end_proof_action=True,
        max_derived_per_state=120,
    )
    base_engine.index_manager = im
    
    vec_engine = UnificationEngineVectorized.from_base_engine(
        base_engine,
        max_fact_pairs=None,
        max_rule_pairs=None,
        padding_atoms=None,
    )
    
    pad = im.padding_idx
    
    # Compare with query 0
    q = dh.test_queries[0]
    query_tensor = im.atom_to_tensor(q.predicate, q.args[0], q.args[1]).unsqueeze(0).to(device)
    query_padded = torch.full((1, 6, 3), im.padding_idx, dtype=torch.long, device=device)
    query_padded[0, 0] = query_tensor[0]
    
    env_orig = BatchedEnv(
        batch_size=1,
        queries=query_padded,
        labels=torch.ones(1, dtype=torch.long, device=device),
        query_depths=torch.ones(1, dtype=torch.long, device=device),
        unification_engine=base_engine,
        mode='eval',
        max_depth=20,
        memory_pruning=False,
        use_exact_memory=False,
        skip_unary_actions=False,
        end_proof_action=True,
        reward_type=0,
        padding_atoms=6,
        padding_states=120,
        true_pred_idx=im.predicate_str2idx.get('True'),
        false_pred_idx=im.predicate_str2idx.get('False'),
        end_pred_idx=im.predicate_str2idx.get('Endf'),
        verbose=0,
        prover_verbose=0,
        device=device,
        runtime_var_start_index=im.constant_no + 1,
        total_vocab_size=im.constant_no + 100,
    )
    
    slot_lengths = torch.ones(1, dtype=torch.long, device=device)
    env_orig.set_eval_dataset(
        queries=query_padded,
        labels=torch.ones(1, dtype=torch.long, device=device),
        query_depths=torch.ones(1, dtype=torch.long, device=device),
        per_slot_lengths=slot_lengths,
    )
    env_orig.reset()
    
    # Now compare raw unification output at each step
    print("Comparing ORIGINAL env derived counts with RAW vectorized:")
    
    current = query_padded.clone()
    next_var = torch.tensor([im.constant_no + 1], dtype=torch.long, device=device)
    excluded = query_padded[:, :1, :]
    
    orig_done = False
    for step in range(8):
        if orig_done:
            break
        
        # Original 
        orig_count = env_orig.derived_states_counts[0].item()
        orig_derived = env_orig.derived_states_batch[0]
        orig_atom_counts = (orig_derived[:int(orig_count), :, 0] != pad).sum(dim=1) if orig_count > 0 else torch.tensor([])
        
        # Vectorized RAW (before any filtering in env)
        vec_derived, vec_count, _ = vec_engine.get_derived_states_compiled(
            current, next_var, excluded
        )
        vec_atom_counts = (vec_derived[0, :vec_count[0], :, 0] != pad).sum(dim=1) if vec_count[0] > 0 else torch.tensor([])
        
        print(f"\nStep {step}:")
        print(f"  Orig: count={orig_count}, max_atoms={orig_atom_counts.max().item() if len(orig_atom_counts) > 0 else 0}")
        print(f"  Vec:  count={vec_count[0].item()}, max_atoms={vec_atom_counts.max().item() if len(vec_atom_counts) > 0 else 0}")
        
        # Check how many exceed 6 in vectorized
        vec_over_6 = (vec_atom_counts > 6).sum().item() if len(vec_atom_counts) > 0 else 0
        orig_over_6 = (orig_atom_counts > 6).sum().item() if len(orig_atom_counts) > 0 else 0
        
        if vec_over_6 > 0 or orig_over_6 > 0:
            print(f"  Vec atoms>6: {vec_over_6}, Orig atoms>6: {orig_over_6}")
        
        # Step original
        action = torch.tensor([0], dtype=torch.long, device=device)
        td = TensorDict({"action": action}, batch_size=[1])
        obs = env_orig.step(td)
        orig_done = obs['next', 'terminated'][0].item() or obs['next', 'truncated'][0].item()
        
        # Update current for vectorized
        if not orig_done and orig_count > 0:
            current = env_orig.current_queries.unsqueeze(0).clone()[:, :, :6, :]
            if current.shape[2] < 6:
                pad_a = torch.full((1, current.shape[1], 6-current.shape[2], 3), pad, dtype=torch.long, device=device)
                current = torch.cat([current, pad_a], dim=2)
            current = current[:, 0:1, :, :].squeeze(1).unsqueeze(0)
            if current.shape[2] < 6:
                pad_a = torch.full((1, 6-current.shape[2], 3), pad, dtype=torch.long, device=device)
                current = torch.cat([current, pad_a], dim=1)

if __name__ == '__main__':
    main()
