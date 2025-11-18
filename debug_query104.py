"""Debug query 104 to see the exact BloomFilter issue"""
import torch
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "str_based"))

from test_envs.test_env_sb3 import setup_sb3_env
from test_envs.test_env_tensor import setup_tensor_env
from test_envs.test_all_configs import prepare_queries
from str_based.str_dataset import Term as StrTerm

def debug_query_104():
    """Debug query 104 which has the mismatch"""
    
    # Get all queries
    queries = prepare_queries(dataset="family", n_queries=400, seed=42)
    
    # Query 104
    query_idx, (pred, head, tail) = queries[104]
    
    print("="*80)
    print(f"Query 104: {pred}({head}, {tail})")
    print("="*80)
    
    config = SimpleNamespace(
        dataset="family",
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
        seed=42,
        max_depth=20,
        use_exact_memory=False,  # BloomFilter
    )
    
    # Setup sb3_env (uses ExactMemory)
    print("\n--- STR_ENV (ExactMemory) ---")
    str_env, _, _ = setup_sb3_env(dataset="family", seed=42, config=config)
    
    q_str = StrTerm(predicate=pred, args=(head, tail))
    label = 1
    str_env.current_query = q_str
    str_env.current_label = label
    str_env.current_query_depth_value = None
    str_obs, _ = str_env._reset([q_str], label)
    
    # Take 5 steps
    for step in range(6):
        num_actions = str_obs['action_mask'].sum().item()
        print(f"Step {step}: memory_size={len(str_env.memory)}, num_actions={num_actions}")
        
        if num_actions == 0 or step == 5:
            break
        
        # Take first action
        str_obs, reward, done, trunc, info = str_env.step(0)
        
        if done or trunc:
            break
    
    # Setup tensor_env (uses BloomFilter)
    print("\n--- TENSOR_ENV (BloomFilter) ---")
    batch_env, debug_helper, _, im_batched, _ = setup_tensor_env(
        dataset="family",
        base_path="./data/",
        seed=42,
        batch_size=1,
        config=config
    )
    
    device = batch_env._device
    query_atom = batch_env.unification_engine.index_manager.atom_to_tensor(pred, head, tail)
    query_padded = torch.full((1, batch_env.padding_atoms, 3), batch_env.padding_idx,
                               dtype=torch.long, device=device)
    query_padded[0, 0] = query_atom
    
    label = 1
    batch_env._all_queries_padded = query_padded
    batch_env._all_labels = torch.tensor([label], dtype=torch.long, device=device)
    batch_env._all_depths = torch.tensor([1], dtype=torch.long, device=device)
    batch_env._all_first_atoms = query_atom.unsqueeze(0)
    batch_env._num_all = 1
    
    # Switch to eval mode (this is what the test does)
    batch_env.set_eval_dataset(
        queries=query_padded,
        labels=torch.tensor([label], dtype=torch.long, device=device),
        query_depths=torch.tensor([1], dtype=torch.long, device=device),
        per_slot_lengths=torch.tensor([1], dtype=torch.long, device=device)  # 1 query in slot 0
    )
    
    td_reset = batch_env.reset()
    
    # Take 5 steps
    from tensordict import TensorDict
    for step in range(6):
        num_actions = td_reset['action_mask'][0].sum().item()
        
        # BloomFilter doesn't expose size, just show actions
        print(f"Step {step}: BloomFilter, num_actions={num_actions}")
        
        if num_actions == 0 or step == 5:
            break
        
        # Take first action
        actions = torch.tensor([0], dtype=torch.long)
        td_reset["action"] = actions
        td_next = batch_env.step(td_reset)
        td_reset = td_next['next']
        
        done_t = td_next['next']['done'][0].item()
        if done_t:
            break
    
    print("\n" + "="*80)
    print("RESULT:")
    print(f"STR_ENV (ExactMemory): {num_actions} actions at step 5")
    print(f"TENSOR_ENV (BloomFilter): should match if BloomFilter is working correctly")

if __name__ == "__main__":
    debug_query_104()
