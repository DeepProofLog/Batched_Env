"""
Profile environment creation to find bottlenecks.
"""

import torch
import time
from utils import Term, Rule
from index_manager import IndexManager
from dataset import DataHandler
from batched_env import BatchedVecEnv


def profile_env_creation():
    """Profile the environment creation process."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("=" * 80)
    print("Profiling Environment Creation")
    print("=" * 80)
    
    t_start = time.time()
    
    # Load data
    print("\n[1/6] Loading dataset...")
    t0 = time.time()
    data_handler = DataHandler(
        dataset_name='wn18rr',
        base_path='data',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt'
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    
    # Create index manager
    print("\n[2/6] Creating IndexManager...")
    t0 = time.time()
    index_manager = IndexManager(
        constants=data_handler.constants,
        predicates=data_handler.predicates,
        max_total_vars=10,
        rules=data_handler.rules,
        device=device
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    
    # Prepare queries
    print("\n[3/6] Preparing queries...")
    t0 = time.time()
    train_queries = data_handler.train_queries[:100]  # Sample
    train_depths = data_handler.train_queries_depths[:100]
    # Create labels (1 for all training queries - they're positive examples)
    train_labels = [1] * len(train_queries)
    print(f"  Time: {time.time() - t0:.2f}s")
    
    # Create environment
    print("\n[4/6] Creating BatchedVecEnv...")
    t0 = time.time()
    env = BatchedVecEnv(
        batch_size=4,
        index_manager=index_manager,
        data_handler=data_handler,
        queries=train_queries,
        labels=train_labels,
        query_depths=train_depths,
        facts=data_handler.facts,
        mode='train',
        max_depth=10,
        memory_pruning=False,
        padding_atoms=10,
        padding_states=20,
        verbose=1,  # Enable profiling output
        prover_verbose=0,
        device=device,
    )
    print(f"  Time: {time.time() - t0:.2f}s")
    
    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"TOTAL TIME: {total_time:.2f}s")
    print("=" * 80)
    
    # Now profile reset
    print("\n\nProfiling reset()...")
    t0 = time.time()
    tensordict = env.reset()
    print(f"  Time: {time.time() - t0:.2f}s")
    
    # Profile a few steps
    print("\n\nProfiling 5 steps...")
    t0 = time.time()
    for i in range(5):
        actions = torch.randint(0, 20, (4,), device=device)
        tensordict['action'] = actions
        tensordict = env.step(tensordict)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {5 / elapsed:.2f}")


if __name__ == '__main__':
    profile_env_creation()
