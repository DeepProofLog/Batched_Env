"""Test script to debug kosovo negative sampling"""
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
    print("INITIALIZING SAMPLER")
    print("="*80)
    
    sampler = get_sampler(
        data_handler=data_handler,
        index_manager=index_manager,
        corruption_scheme=["tail"],
        device=device
    )

    print("\n" + "="*80)
    print("TESTING NEGATIVE GENERATION FOR locatedInCR(kosovo, europe)")
    print("="*80)

    # Create the query state for locatedInCR(kosovo, europe)
    query = Term("locatedInCR", ["kosovo", "europe"])
    
    # Convert to tensor format
    kosovo_idx = index_manager.constant_str2idx["kosovo"]
    locatedInCR_idx = index_manager.predicate_str2idx["locatedInCR"]
    europe_idx = index_manager.constant_str2idx["europe"]
    
    # Create state tensor (shape: 1, padding_atoms, max_arity+1)
    padding_atoms = 4
    max_arity = 2
    pos_subs = torch.full((1, padding_atoms, max_arity + 1), 
                          index_manager.padding_idx, 
                          dtype=torch.int32)
    
    # Fill in the query at position 0
    pos_subs[0, 0, 0] = locatedInCR_idx  # relation
    pos_subs[0, 0, 1] = kosovo_idx       # head
    pos_subs[0, 0, 2] = europe_idx       # tail
    
    print(f"\nQuery: locatedInCR(kosovo, europe)")
    print(f"Indices: ({kosovo_idx}, {locatedInCR_idx}, {europe_idx})")
    
    # Get negatives using num_negs=None (enumerate all)
    print("\n" + "="*80)
    print("CALLING get_negatives WITH num_negs=None (enumerate all)")
    print("="*80)
    
    neg_subs = sampler.get_negatives(
        sub_indices=pos_subs,
        padding_atoms=padding_atoms,
        max_arity=max_arity,
        device=device,
        num_negs=None,  # Enumerate all
        debug=True
    )
    
    print(f"\n" + "="*80)
    print(f"FINAL RESULT")
    print("="*80)
    print(f"Shape of neg_subs: {neg_subs.shape}")
    print(f"Number of negatives returned: {neg_subs.shape[1]}")
    
    # Decode and display the negatives
    print(f"\nNegatives:")
    for i in range(neg_subs.shape[1]):
        r_idx = neg_subs[0, i, 0, 0].item()
        h_idx = neg_subs[0, i, 0, 1].item()
        t_idx = neg_subs[0, i, 0, 2].item()
        
        if t_idx == index_manager.padding_idx:
            break
            
        h_str = index_manager.constant_idx2str.get(h_idx, f"<{h_idx}>")
        r_str = index_manager.predicate_idx2str.get(r_idx, f"<{r_idx}>")
        t_str = index_manager.constant_idx2str.get(t_idx, f"<{t_idx}>")
        print(f"  [{i}] {r_str}({h_str}, {t_str})")
    
    # Check specifically for asia
    asia_idx = index_manager.constant_str2idx.get("asia")
    if asia_idx:
        asia_found = False
        for i in range(neg_subs.shape[1]):
            t_idx = neg_subs[0, i, 0, 2].item()
            if t_idx == asia_idx:
                asia_found = True
                break
        
        if asia_found:
            print(f"\n✓ 'asia' (idx={asia_idx}) WAS found in negatives")
        else:
            print(f"\n✗ 'asia' (idx={asia_idx}) was NOT found in negatives")
            print(f"\nExpected continents and their status:")
            for cont in ["asia", "europe", "africa", "americas", "oceania"]:
                cont_idx = index_manager.constant_str2idx.get(cont)
                if cont_idx:
                    found = any(neg_subs[0, i, 0, 2].item() == cont_idx 
                               for i in range(neg_subs.shape[1]))
                    status = "FOUND" if found else "MISSING"
                    symbol = "✓" if found else "✗"
                    print(f"    {symbol} {cont:10s} (idx={cont_idx:3d}): {status}")

finally:
    # Cleanup
    if os.path.exists(facts_file_path):
        os.unlink(facts_file_path)
