"""Test what negatives are actually generated for kosovo"""
from dataset import DataHandler
from index_manager import IndexManager
from neg_sampling import get_sampler
import tempfile
import os
import torch

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
sampler = get_sampler(
    data_handler=data_handler,
    index_manager=index_manager,
    corruption_scheme=["tail"],
    device=device
)

# Get indices
kosovo_idx = index_manager.constant_str2idx["kosovo"]
locatedInCR_idx = index_manager.predicate_str2idx["locatedInCR"]
europe_idx = index_manager.constant_str2idx["europe"]
asia_idx = index_manager.constant_str2idx["asia"]

print(f"\n=== INDICES ===")
print(f"kosovo: {kosovo_idx}")
print(f"locatedInCR: {locatedInCR_idx}")
print(f"europe: {europe_idx}")
print(f"asia: {asia_idx}")

# Check domain info
print(f"\n=== DOMAIN INFO ===")
europe_domain = data_handler.entity2domain.get("europe")
asia_domain = data_handler.entity2domain.get("asia")
kosovo_domain = data_handler.entity2domain.get("kosovo")
print(f"europe domain: {europe_domain}")
print(f"asia domain: {asia_domain}")
print(f"kosovo domain: {kosovo_domain}")

if europe_domain:
    entities_in_europe_domain = data_handler.domain2entity[europe_domain]
    print(f"\nEntities in '{europe_domain}' domain ({len(entities_in_europe_domain)} total):")
    print(f"  {entities_in_europe_domain}")

# Create a positive batch with locatedInCR(kosovo, europe)
positive_batch = torch.tensor([[kosovo_idx, locatedInCR_idx, europe_idx]], dtype=torch.int64)
print(f"\n=== POSITIVE BATCH ===")
print(f"Triple (h,r,t): ({kosovo_idx}, {locatedInCR_idx}, {europe_idx})")
print(f"Triple (str): locatedInCR(kosovo, europe)")

# Call corrupt_batch_all to see what negatives are generated BEFORE filtering
print(f"\n=== NEGATIVES BEFORE FILTERING ===")
neg_batches = sampler.corrupt_batch_all(positive_batch)
print(f"Number of negatives generated: {neg_batches[0].shape[0]}")
print(f"Negatives:")
for neg in neg_batches[0]:
    h, r, t = neg.tolist()
    h_str = index_manager.constant_idx2str.get(h, f"<{h}>")
    r_str = index_manager.predicate_idx2str.get(r, f"<{r}>")
    t_str = index_manager.constant_idx2str.get(t, f"<{t}>")
    print(f"  {r_str}({h_str}, {t_str})")

# Now apply filter
print(f"\n=== NEGATIVES AFTER FILTERING ===")
if neg_batches[0].numel() > 0:
    mask = sampler.filterer(neg_batches[0])
    filtered = neg_batches[0][mask]
    print(f"Number of negatives after filtering: {filtered.shape[0]}")
    print(f"Filtered negatives:")
    for neg in filtered:
        h, r, t = neg.tolist()
        h_str = index_manager.constant_idx2str.get(h, f"<{h}>")
        r_str = index_manager.predicate_idx2str.get(r, f"<{r}>")
        t_str = index_manager.constant_idx2str.get(t, f"<{t}>")
        print(f"  {r_str}({h_str}, {t_str})")

# Cleanup
if os.path.exists(facts_file_path):
    os.unlink(facts_file_path)
