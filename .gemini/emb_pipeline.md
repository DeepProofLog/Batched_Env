Sequential Pipeline: From Indices to Logits
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Raw Indices [B, S, A, 3]                                            │
│   Each atom = (predicate_idx, const1_idx, const2_idx)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TransE Atom Embedding [B, S, A, E]                                  │
│   atom_emb = pred_emb + (const1_emb - const2_emb)                           │
│   Norm: ~27 (sum of 3 vectors with std=1)                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Mean State Aggregation [B, S, E]                                    │
│   state_emb = mean(atom_emb, dim=atoms)                                     │
│   Norm: 27 / √6 ≈ 11                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: SharedBody (8 Residual Layers) [B, S, H]                            │
│   x = Linear(x)                                                              │
│   for 8 layers: x = x + f(x)  ← Residuals accumulate!                       │
│   Norm: 11 → 48 (3-4x amplification)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: PolicyHead (Linear) [B, S, E]                                       │
│   encoded = Linear(shared_out)                                               │
│   Norm: 48 → 11 (compression via Xavier init)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Dot Product [B, S]                                                  │
│   logits = obs @ actions.T                                                   │
│   = ||obs|| × ||action|| × cos(θ)                                           │
│   At init: ≈ 11 × 11 × 0 = 0 (random directions)                            │
│   Trained: ≈ 11 × 11 × ±1 = ±121                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Attention Scaling                                                   │
│   logits = logits / √250 = logits / 15.8                                    │
│   Trained: ±121 / 15.8 = ±7.7                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓  
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: Temperature Scaling                                                 │
│   logits = logits / temperature                                              │
│   With temp=10: ±7.7 / 10 = ±0.77                                           │
│   This keeps entropy reasonable!                                             │
└─────────────────────────────────────────────────────────────────────────────┘




High Temperature Consequences
Issue	                            Effect                          Severity
Smaller Gradients	∇log π ∝ 1/temp → 10x slower policy updates	    ⚠️ High
Credit Assignment	Can't distinguish which action was good	        ⚠️ High
Effective LR	    lr=5e-5 with temp=10 ≈ effective lr=5e-6	    ⚠️ High
Exploration	        Good early, harmful late	                    Moderate
Value Learning	    Noisy trajectories → slow convergence	        Moderate
