"""
Detailed debugging of BloomFilter membership testing at step 5.
"""
import torch
import numpy as np
from str_based.str_index_manager import IndexManager as StringIndexManager
from str_based.str_env import StringReasoningEnv
from env import BatchedEnv
from data_handler import get_test_queries, set_eval_dataset

# Load dataset
print("Loading dataset...")
from argparse import Namespace
args = Namespace(dataset='family', num_step=10, skip_unary_actions=True, end_proof_action=True)

str_idx = StringIndexManager(args.dataset)
str_idx.load_from_file()

train_queries, valid_queries, test_queries = get_test_queries(args.dataset, str_idx)
set_eval_dataset(test_queries, 400)

# Query 104
query_idx = 104
query = test_queries[query_idx]
print(f"Query {query_idx}: {str_idx.get_str_fact(query)}")
print()

# STR_ENV with ExactMemory
print("=== STR_ENV with ExactMemory ===")
str_env = StringReasoningEnv(
    str_idx, args.num_step, 
    padding_states=40,
    memory_pruning=True,
    eval_pruning=True,
    skip_unary_actions=args.skip_unary_actions,
    end_proof_action=args.end_proof_action
)
str_env.set_query(query)

for step in range(6):
    print(f"\nStep {step}:")
    print(f"  current_queries: {[str_idx.get_str_fact(q) for q in str_env.current_queries if q[0] != str_idx.padding_idx]}")
    print(f"  Memory size: {len(str_env.memory)}")
    
    actions = str_env.get_available_actions()
    print(f"  Available actions: {len(actions)}")
    
    if step < 5:
        # Take first action
        next_state, reward, done = str_env.step(actions[0])
print()

# BatchedEnv with BloomFilter
print("=== BatchedEnv with BloomFilter ===")
device = torch.device('cpu')
tensor_env = BatchedEnv(
    dataset=args.dataset,
    num_envs=1,
    num_step=args.num_step,
    device=device,
    padding_states=40,
    memory_pruning=True,
    eval_pruning=True,
    use_exact_memory=False,  # Use BloomFilter
    memory_bits_pow=22,
    memory_hashes=7,
    skip_unary_actions=args.skip_unary_actions,
    end_proof_action=args.end_proof_action
)

# Set query
queries_tensor = torch.tensor([[query]], device=device)
tensor_env.reset(queries_tensor)

idx_mgr = tensor_env.index_manager

for step in range(6):
    print(f"\nStep {step}:")
    
    # Show current queries
    current = tensor_env.tensordict['current_queries'][0]  # [M, 3]
    current_list = []
    for i in range(current.shape[0]):
        atom = current[i].tolist()
        if atom[0] != tensor_env.index_manager.padding_idx:
            current_list.append(idx_mgr.decode_atom(atom))
    print(f"  current_queries: {current_list}")
    
    # Get actions
    action_mask = tensor_env.tensordict['action_mask'][0]  # [A]
    num_actions = action_mask.sum().item()
    print(f"  Available actions: {num_actions}")
    
    # Show memory membership for current_queries
    if step == 5:
        print("\n  === DETAILED MEMBERSHIP CHECK AT STEP 5 ===")
        
        # Check each atom in current_queries
        for i in range(current.shape[0]):
            atom = current[i].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 3]
            if current[i, 0].item() != tensor_env.index_manager.padding_idx:
                is_visited = tensor_env.memory_backend.membership(
                    atom,
                    torch.tensor([0], device=device)
                )[0, 0].item()
                atom_str = idx_mgr.decode_atom(current[i].tolist())
                print(f"    {atom_str}: visited={is_visited}")
        
        # Now check the entire state
        full_state = current.unsqueeze(0).unsqueeze(0)  # [1, 1, M, 3]
        is_state_visited = tensor_env.memory_backend.membership(
            full_state,
            torch.tensor([0], device=device)
        )[0, 0].item()
        print(f"\n  Full state visited: {is_state_visited}")
        print(f"  (This should be True if any duplicate from steps 0-4)")
    
    if step < 5:
        # Take first available action
        action_idx = torch.where(action_mask)[0][0]
        tensor_env.step(action_idx.unsqueeze(0))
