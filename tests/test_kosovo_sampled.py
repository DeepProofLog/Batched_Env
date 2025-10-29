"""Test kosovo negative sampling with num_negs=4 (sampled mode)"""
import torch
from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler
from utils import Term
import tempfile
import os

# Create empty facts file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    facts_file_path = f.name

try:
    # Load the dataset
    base_path = os.path.join(os.path.dirname(__file__), "data")
    data_handler = DataHandler(
        dataset_name="countries_s3",
        base_path=base_path,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        rules_file="rules.txt",
        facts_file=facts_file_path,
        n_train_queries=None,
        n_eval_queries=None,
        n_test_queries=None,
        prob_facts=True,
        topk_facts=None,
        topk_facts_threshold=None,
        corruption_mode="static",
    )

    # Create index manager
    index_manager = IndexManager(
        data_handler.constants,
        data_handler.predicates,
        max_total_vars=10,
        rules=data_handler.rules,
        max_arity=data_handler.max_arity,
        device="cpu",
        padding_atoms=4,
    )
    index_manager.build_fact_index(data_handler.facts)

    # Get sampler
    device = torch.device("cpu")
    print("\n" + "="*80)
    print("TESTING WITH num_negs=4 (SAMPLED MODE)")
    print("="*80)
    
    sampler = get_sampler(
        data_handler=data_handler,
        index_manager=index_manager,
        corruption_scheme=["tail"],
        device=device
    )

    # Create the query state for locatedInCR(kosovo, europe)
    kosovo_idx = index_manager.constant_str2idx["kosovo"]
    locatedInCR_idx = index_manager.predicate_str2idx["locatedInCR"]
    europe_idx = index_manager.constant_str2idx["europe"]
    
    # Create state tensor
    padding_atoms = 4
    max_arity = 2
    pos_subs = torch.full((1, padding_atoms, max_arity + 1), 
                          index_manager.padding_idx, 
                          dtype=torch.int32)
    
    pos_subs[0, 0, 0] = locatedInCR_idx  
    pos_subs[0, 0, 1] = kosovo_idx       
    pos_subs[0, 0, 2] = europe_idx       
    
    print(f"\nQuery: locatedInCR(kosovo, europe)")
    
    # Test multiple times to see if it's random
    print("\nRunning 10 trials with num_negs=4:")
    asia_idx = index_manager.constant_str2idx.get("asia")
    asia_count = 0
    
    for trial in range(10):
        neg_subs = sampler.get_negatives(
            sub_indices=pos_subs,
            padding_atoms=padding_atoms,
            max_arity=max_arity,
            device=device,
            num_negs=4,  # Sampled mode
            debug=False   # Turn off debug for cleaner output
        )
        
        # Check if asia is in this trial
        asia_found = False
        negatives = []
        for i in range(neg_subs.shape[1]):
            t_idx = neg_subs[0, i, 0, 2].item()
            if t_idx == index_manager.padding_idx:
                break
            t_str = index_manager.constant_idx2str.get(t_idx, f"<{t_idx}>")
            negatives.append(t_str)
            if t_idx == asia_idx:
                asia_found = True
                asia_count += 1
        
        status = "✓ HAS asia" if asia_found else "✗ NO asia"
        print(f"  Trial {trial+1}: {status:15s} | Negatives: {negatives}")
    
    print(f"\nSummary: asia appeared in {asia_count}/10 trials")
    
    if asia_count < 10:
        print("\n⚠ WARNING: asia is not appearing in all trials!")
        print("This suggests the sampling is random and may sometimes exclude asia.")
        print("With only 4 negatives available and requesting 4, we should get all of them.")

finally:
    if os.path.exists(facts_file_path):
        os.unlink(facts_file_path)
