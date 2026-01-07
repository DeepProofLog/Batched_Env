#!/usr/bin/env python3
"""Test script to verify KGE model loading and evaluation."""
import sys
sys.path.insert(0, '/home/castellanoontiv/Batched_env/kge_experiments/kge_pytorch')

from model_torch import build_model
from train_torch import evaluate_sampled
import torch
import json

def main():
    model_dir = '/home/castellanoontiv/Batched_env/kge_experiments/kge_pytorch/models/torch_family_complex_v5'
    
    # Load config
    with open(f'{model_dir}/config.json') as f:
        config = json.load(f)
    with open(f'{model_dir}/entity2id.json') as f:
        entity2id = json.load(f)
    with open(f'{model_dir}/relation2id.json') as f:
        relation2id = json.load(f)
    
    print(f"Config: {config}")
    print(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")
    
    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(
        name=config['model'],
        num_entities=config['num_entities'],
        num_relations=config['num_relations'],
        dim=config['dim']
    )
    
    # Load weights (strip _orig_mod prefix from compiled models)
    state_dict = torch.load(f'{model_dir}/weights.pth', map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[10:] if k.startswith('_orig_mod.') else k
        new_state_dict[new_k] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load test data
    def load_triples(path):
        triples = []
        with open(path) as f:
            for line in f:
                parts = line.strip().replace('(', ' ').replace(')', ' ').replace(',', ' ').split()
                if len(parts) >= 3:
                    rel, h, t = parts[0], parts[1], parts[2]
                    if rel in relation2id and h in entity2id and t in entity2id:
                        triples.append((entity2id[h], relation2id[rel], entity2id[t]))
        return triples
    
    data_dir = '/home/castellanoontiv/Batched_env/kge_experiments/data/family'
    test_triples = load_triples(f'{data_dir}/test.txt')
    train_triples = load_triples(f'{data_dir}/train.txt')
    valid_triples = load_triples(f'{data_dir}/valid.txt')
    
    known_facts = set(train_triples + valid_triples + test_triples)
    print(f"Test: {len(test_triples)}, Known facts: {len(known_facts)}")
    
    # Evaluate with sampled negatives (like paper)
    print("\nRunning sampled evaluation (100 negatives)...")
    metrics = evaluate_sampled(
        model=model,
        triples=test_triples,
        num_entities=len(entity2id),
        known_facts=known_facts,
        device=device,
        num_negatives=100,
        verbose=True
    )
    
    print(f"\n{'='*50}")
    print("KGE-ONLY RESULTS (100 negatives, full test set)")
    print(f"{'='*50}")
    print(f"MRR:      {metrics['MRR']:.4f}")
    print(f"Hits@1:   {metrics['Hits@1']:.4f}")
    print(f"Hits@3:   {metrics['Hits@3']:.4f}")
    print(f"Hits@10:  {metrics['Hits@10']:.4f}")

if __name__ == "__main__":
    main()
