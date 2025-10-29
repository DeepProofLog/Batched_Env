"""Debug the domain pool for regions"""
import torch

# Simulate what happens during domain pool creation
# Based on the actual domain file content

domain2entity = {
    'regions': ['oceania', 'asia', 'europe', 'africa', 'americas'],
    'subregions': ['northern_america', 'eastern_europe', 'australia_and_new_zealand', 
                   'melanesia', 'micronesia', 'eastern_africa', 'southern_asia', 
                   'eastern_asia', 'south_america', 'central_europe', 'western_asia',
                   'northern_africa', 'western_africa', 'northern_europe', 'middle_africa',
                   'caribbean', 'polynesia', 'western_europe', 'southern_europe',
                   'central_america', 'southern_africa', 'central_asia', 'south_eastern_asia'],
    # ... countries would be here
}

# Simulate the index mapping (these are made-up values for illustration)
# In reality, these come from index_manager.constant_str2idx
constant_str2idx = {
    'oceania': 164,
    'asia': 15,
    'europe': 71,
    'africa': 2,
    'americas': 7,
    'kosovo': 116,
}

print("=== SIMULATED DOMAIN POOL CREATION ===\n")

# Build domain2idx
domain_name = 'regions'
entities = domain2entity[domain_name]
print(f"Entities in '{domain_name}' domain: {entities}")

# This is what happens at line 745 in neg_sampling.py
try:
    domain_indices = [constant_str2idx[e] for e in entities]
    print(f"Their indices: {domain_indices}")
except KeyError as e:
    print(f"ERROR: Entity '{e.args[0]}' not found in constant_str2idx!")
    print("This would cause the sampler initialization to fail.")
    
    # Show which entities ARE indexed
    indexed = [e for e in entities if e in constant_str2idx]
    not_indexed = [e for e in entities if e not in constant_str2idx]
    print(f"\nIndexed entities: {indexed}")
    print(f"Not indexed entities: {not_indexed}")
    
    # If code continues with partial list
    domain_indices = [constant_str2idx[e] for e in indexed]
    print(f"\nIf we skip missing entities, indices would be: {domain_indices}")

print("\n=== SIMULATION OF corrupt_batch_all ===\n")

# Simulate what happens when corrupting locatedInCR(kosovo, europe)
europe_idx = constant_str2idx.get('europe', None)
if europe_idx is None:
    print("ERROR: europe not indexed!")
else:
    print(f"Original tail: europe (idx={europe_idx})")
    
    # Create the pool (simplified - in reality this is in domain_padded tensor)
    pool = torch.tensor(domain_indices, dtype=torch.int32)
    print(f"Pool (all entities in regions domain): {pool.tolist()}")
    
    # Exclude the original entity
    e = europe_idx
    cand = pool[pool != e]
    print(f"Candidates after excluding europe ({e}): {cand.tolist()}")
    
    # Map back to names
    idx2str = {v: k for k, v in constant_str2idx.items()}
    cand_names = [idx2str.get(int(i), f"<{i}>") for i in cand.tolist()]
    print(f"Candidate names: {cand_names}")
    
    if 15 in cand.tolist():
        print("\n✓ asia (idx=15) IS in the candidates")
    else:
        print("\n✗ asia (idx=15) is NOT in the candidates")
        print("  Possible reasons:")
        print("  1. asia was not added to domain_indices")
        print("  2. asia has the same index as europe (unlikely)")
        print("  3. There's a bug in the pool filtering logic")
