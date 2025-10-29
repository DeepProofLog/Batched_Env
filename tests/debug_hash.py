"""Check for hash collision in the filter"""
from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler, SortedHashTripleFilter
import tempfile
import os
import torch
import numpy as np

# Create empty facts file
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    facts_file_path = f.name

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

# Convert all_known_triples to mapped format
np_facts = np.array([[f.args[0], f.predicate, f.args[1]] for f in data_handler.all_known_triples], dtype=str)
from pykeen.triples import TriplesFactory
triples_factory = TriplesFactory.from_labeled_triples(
    triples=np_facts,
    entity_to_id=index_manager.constant_str2idx,
    relation_to_id=index_manager.predicate_str2idx,
    compact_id=False,
    create_inverse_triples=False
)
mapped_triples = triples_factory.mapped_triples

# Create filter
filterer = SortedHashTripleFilter(mapped_triples)

# Get indices
kosovo_idx = index_manager.constant_str2idx.get("kosovo")
locatedInCR_idx = index_manager.predicate_str2idx.get("locatedInCR")
continents = ["asia", "europe", "africa", "americas", "oceania"]
continent_indices = {c: index_manager.constant_str2idx.get(c) for c in continents}

print(f"\n=== INDICES ===")
print(f"kosovo: {kosovo_idx}")
print(f"locatedInCR: {locatedInCR_idx}")
for c, idx in continent_indices.items():
    print(f"{c}: {idx}")

print(f"\n=== HASH VALUES ===")
for c, cont_idx in continent_indices.items():
    if cont_idx is not None:
        # Compute hash as the filter does: (h << 42) | (r << 21) | t
        h = int(kosovo_idx) if kosovo_idx is not None else 0
        r = int(locatedInCR_idx) if locatedInCR_idx is not None else 0
        t = int(cont_idx)
        hash_val = (h << 42) | (r << 21) | t
        print(f"locatedInCR(kosovo,{c:10s}): hash={hash_val}")

print(f"\n=== HASHES IN FILTER ===")
print(f"Total unique hashes in filter: {filterer._hashes_sorted.numel()}")
print(f"Sample hashes: {filterer._hashes_sorted[:10].tolist()}")

print(f"\n=== FILTER TEST ===")
for c, cont_idx in continent_indices.items():
    if cont_idx is not None and kosovo_idx is not None and locatedInCR_idx is not None:
        triple = torch.tensor([[kosovo_idx, locatedInCR_idx, cont_idx]], dtype=mapped_triples.dtype)
        is_kept = filterer(triple)[0].item()
        
        # Also check if it's in mapped_triples
        in_mapped = ((mapped_triples[:, 0] == kosovo_idx) &
                    (mapped_triples[:, 1] == locatedInCR_idx) &
                    (mapped_triples[:, 2] == cont_idx)).any().item()
        
        print(f"{c:10s}: kept_by_filter={is_kept:5} (True means NOT filtered), in_mapped_triples={in_mapped}")

#  Check if there's a hash collision
print(f"\n=== CHECKING FOR HASH COLLISIONS ===")
asia_idx = continent_indices["asia"]
if asia_idx is not None and kosovo_idx is not None and locatedInCR_idx is not None:
    h = int(kosovo_idx)
    r = int(locatedInCR_idx)
    t = int(asia_idx)
    target_hash = (h << 42) | (r << 21) | t
    
    # Check if this hash is in the filter
    pos = torch.searchsorted(filterer._hashes_sorted, torch.tensor([target_hash]))
    print(f"Target hash for locatedInCR(kosovo,asia): {target_hash}")
    print(f"Position in sorted hashes: {pos.item()}")
    
    if pos.item() < filterer._hashes_sorted.numel():
        hash_at_pos = filterer._hashes_sorted[pos.item()].item()
        print(f"Hash at that position: {hash_at_pos}")
        if hash_at_pos == target_hash:
            print("  -> MATCH FOUND! This hash IS in the filter")
            # Find which triple this corresponds to
            for i, triple in enumerate(mapped_triples):
                triple_hash = (int(triple[0].item()) << 42) | (int(triple[1].item()) << 21) | int(triple[2].item())
                if triple_hash == target_hash:
                    h_str = index_manager.constant_idx2str[triple[0].item()]
                    r_str = index_manager.predicate_idx2str[triple[1].item()]
                    t_str = index_manager.constant_idx2str[triple[2].item()]
                    print(f"  -> Corresponds to: {r_str}({h_str},{t_str})")
        else:
            print("  -> No match, hash not in filter")

# Cleanup
if os.path.exists(facts_file_path):
    os.unlink(facts_file_path)
