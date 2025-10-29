"""Debug script to check what's in the triple filter for kosovo"""
import torch
from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler
import tempfile
import os

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
index_manager = IndexManager(data_handler)

# Get the sampler
device = torch.device("cpu")
sampler = get_sampler(
    data_handler=data_handler,
    index_manager=index_manager,
    corruption_scheme=["tail"],
    device=device
)

print(f"\nTotal triples in filter: {len(sampler.mapped_triples)}")
print(f"Sample of mapped triples shape: {sampler.mapped_triples.shape}")

# Check what's in all_known_triples
print(f"\nTotal facts: {len(data_handler.facts)}")
print(f"Total train queries: {len(data_handler.train_queries)}")
print(f"Total valid queries: {len(data_handler.valid_queries)}")
print(f"Total test queries: {len(data_handler.test_queries)}")
print(f"Total all_known_triples: {len(data_handler.all_known_triples)}")

# Check for kosovo facts
kosovo_idx = index_manager.constant_str2idx.get("kosovo")
locatedInCR_idx = index_manager.predicate_str2idx.get("locatedInCR")
asia_idx = index_manager.constant_str2idx.get("asia")
europe_idx = index_manager.constant_str2idx.get("europe")
africa_idx = index_manager.constant_str2idx.get("africa")
americas_idx = index_manager.constant_str2idx.get("americas")
oceania_idx = index_manager.constant_str2idx.get("oceania")

print(f"\nIndices:")
print(f"  kosovo: {kosovo_idx}")
print(f"  locatedInCR: {locatedInCR_idx}")
print(f"  asia: {asia_idx}")
print(f"  europe: {europe_idx}")
print(f"  africa: {africa_idx}")
print(f"  americas: {americas_idx}")
print(f"  oceania: {oceania_idx}")

# Check all_known_triples for kosovo+locatedInCR
print(f"\nKosovo facts in all_known_triples:")
for triple in data_handler.all_known_triples:
    if triple.args[0] == "kosovo" and triple.predicate == "locatedInCR":
        print(f"  {triple}")

# Check if locatedInCR(kosovo,asia) is in mapped_triples
if kosovo_idx is not None and locatedInCR_idx is not None and asia_idx is not None:
    # mapped_triples is (h, r, t) format
    target = torch.tensor([[kosovo_idx, locatedInCR_idx, asia_idx]], dtype=sampler.mapped_triples.dtype)
    
    # Check if it's in the filter
    is_filtered = sampler.filterer(target)
    print(f"\nIs locatedInCR(kosovo,asia) kept by filter? {is_filtered[0].item()}")
    
    # Check if it's in mapped_triples directly
    matches = (sampler.mapped_triples[:, 0] == kosovo_idx) & \
              (sampler.mapped_triples[:, 1] == locatedInCR_idx) & \
              (sampler.mapped_triples[:, 2] == asia_idx)
    print(f"Is locatedInCR(kosovo,asia) in mapped_triples? {matches.any().item()}")
    
    # Check all kosovo+locatedInCR combinations in mapped_triples
    kosovo_located = (sampler.mapped_triples[:, 0] == kosovo_idx) & \
                     (sampler.mapped_triples[:, 1] == locatedInCR_idx)
    if kosovo_located.any():
        print(f"\nAll kosovo+locatedInCR in mapped_triples:")
        for triple in sampler.mapped_triples[kosovo_located]:
            h, r, t = triple.tolist()
            t_str = index_manager.constant_idx2str.get(t, f"<{t}>")
            print(f"  locatedInCR(kosovo, {t_str})")
    
    # Check all continents
    print(f"\nChecking all continents:")
    for cont_name, cont_idx in [("asia", asia_idx), ("europe", europe_idx), 
                                  ("africa", africa_idx), ("americas", americas_idx),
                                  ("oceania", oceania_idx)]:
        if cont_idx is not None:
            target = torch.tensor([[kosovo_idx, locatedInCR_idx, cont_idx]], 
                                 dtype=sampler.mapped_triples.dtype)
            is_kept = sampler.filterer(target)[0].item()
            
            # Check if in mapped_triples
            in_mapped = ((sampler.mapped_triples[:, 0] == kosovo_idx) & \
                        (sampler.mapped_triples[:, 1] == locatedInCR_idx) & \
                        (sampler.mapped_triples[:, 2] == cont_idx)).any().item()
            
            print(f"  {cont_name:10s}: kept={is_kept}, in_mapped={in_mapped}")

# Cleanup
if os.path.exists(facts_file_path):
    os.unlink(facts_file_path)
