#!/usr/bin/env python3
"""Quick test for PyTorch inference to debug segfault."""

import os
import sys
import torch

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from kge_pytorch.kge_inference_torch import load_model

def main():
    model_dir = "./kge_pytorch/models"
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        print("Loading model...")
        model, entity2id, relation2id, cfg = load_model(model_dir, device)
        print("✓ Model loaded successfully")
        print(f"  Entities: {len(entity2id)}")
        print(f"  Relations: {len(relation2id)}")
        
        print("\nTesting single triple scoring...")
        h = torch.tensor([0], dtype=torch.long, device=device)
        r = torch.tensor([0], dtype=torch.long, device=device)
        t = torch.tensor([0], dtype=torch.long, device=device)
        
        with torch.no_grad():
            score = model.score_triples(h, r, t)
            print(f"✓ Score computed: {score.item():.4f}")
        
        print("\nTesting batch scoring...")
        batch_size = 64
        h_batch = torch.zeros(batch_size, dtype=torch.long, device=device)
        r_batch = torch.zeros(batch_size, dtype=torch.long, device=device)
        t_batch = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        with torch.no_grad():
            scores = model.score_triples(h_batch, r_batch, t_batch)
            print(f"✓ Batch scores computed: {scores.shape}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
