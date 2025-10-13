# Curriculum Learning Visual Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  runner.py: Parse Arguments                                      │
│  ├─ --top_k_curriculum                                          │
│  ├─ --top_k_initial = 20                                        │
│  ├─ --top_k_final = 5                                           │
│  └─ --top_k_schedule = 'linear'                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  train.py: Initialize Model                                      │
│  policy_kwargs = {                                               │
│      'top_k_curriculum': True,                                   │
│      'top_k_initial': 20,                                        │
│      'top_k_final': 5                                            │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  model.py: Policy Initialization                                 │
│  self.top_k_actions = 20  # Set to initial value                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  train.py: Create Curriculum Callback                            │
│  curriculum_cb = TopKCurriculumCallback(                         │
│      initial_k=20,                                               │
│      final_k=5,                                                  │
│      schedule='linear'                                           │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Training Loop Begins                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
         ┌────────────────────┴────────────────────┐
         ↓                                         ↓
┌──────────────────┐                    ┌──────────────────────┐
│  Environment     │                    │  Curriculum          │
│  Steps           │                    │  Callback            │
│                  │                    │  ._on_step()         │
│  1. Get obs      │                    │                      │
│  2. Policy       │←───────────────────│  Updates             │
│     forward()    │  K updated here    │  policy.top_k_actions│
│  3. Filter top-K │                    │  based on progress   │
│  4. Sample action│                    │                      │
│  5. Execute      │                    │                      │
└──────────────────┘                    └──────────────────────┘
         ↓
    Repeat until training completes
```

## Curriculum Progression Visualization

### Linear Schedule (20 → 5)

```
Training Progress:    0%        25%        50%        75%       100%
                      ├──────────┼──────────┼──────────┼──────────┤
                      
Top-K Value:         20         16         12          8          5
                      ●──────────●──────────●──────────●──────────●

Action Space:    ████████   ██████   ████   ██   █
                  Large    Medium   Small  Tiny Minimal
                   ↑                                    ↑
              Full Exploration                    Focused Exploitation
```

### Exponential Schedule (20 → 5)

```
Training Progress:    0%        25%        50%        75%       100%
                      ├──────────┼──────────┼──────────┼──────────┤
                      
Top-K Value:         20         13         10          7          5
                      ●────●───●─●──●─●─●─●●●●●●●●●●●●●●●●●●●●●●●

Action Space:    ████████   █████   ████   ███   ██
                   ↑                                    ↑
              Fast Reduction                      Gradual Stabilization
```

### Step Schedule (20 → 10 → 5)

```
Training Progress:    0%        25%        50%        75%       100%
                      ├──────────┼──────────┼──────────┼──────────┤
                      
Top-K Value:         20         20         10         10          5
                      ●──────────●──────────●──────────●──────────●
                      └─Stage 1──┘└─Stage 2─┘└──────Stage 3───────┘

Action Space:    ████████         ████              ██
                   ↑               ↑                ↑
              Exploration     Moderate Focus    Exploitation
```

## Detailed Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Environment Step (timestep t)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  TopKCurriculumCallback._on_step()                               │
│                                                                  │
│  1. Calculate progress:                                          │
│     progress = t / total_timesteps                               │
│     Example: 500,000 / 1,000,000 = 0.5 (50%)                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. Compute new K value based on schedule:                       │
│                                                                  │
│  Linear:                                                         │
│  K = 20 - 0.5 * (20 - 5) = 20 - 7.5 = 12                       │
│                                                                  │
│  Exponential:                                                    │
│  K = 20 * (5/20)^0.5 = 20 * 0.5 = 10                           │
│                                                                  │
│  Step:                                                           │
│  if progress < 0.5: K = 20                                       │
│  elif progress < 1.0: K = 10                                     │
│  else: K = 5                                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. Check if K changed:                                          │
│     if new_K != current_K:                                       │
│         model.policy.top_k_actions = new_K                       │
│         Log to console and TensorBoard                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  CustomActorCriticPolicy.forward()                               │
│                                                                  │
│  Current state → Value Network → Action Values                   │
│                                                                  │
│  All available actions (e.g., 50 actions):                       │
│  [a1, a2, a3, ..., a50]                                         │
│                                                                  │
│  Value scores:                                                   │
│  [0.8, 0.5, 0.9, ..., 0.1]                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  _filter_action_logits_top_k()                                   │
│                                                                  │
│  Current K = 12 (from curriculum)                                │
│                                                                  │
│  1. Rank actions by value:                                       │
│     Top 12: [a3, a1, a7, a15, ..., a23]                         │
│                                                                  │
│  2. Mask out non-top-K actions:                                  │
│     [a3, a1, a7, ..., a23, -∞, -∞, ..., -∞]                     │
│                                                                  │
│  3. Create probability distribution over top 12 only             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Policy samples action from filtered distribution                │
│  Execute action in environment                                   │
│  Collect reward and next state                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Continue to next timestep
```

## State Transition Example

```
Time: t=0 (Start of training)
Progress: 0%
K = 20 (initial)

State: [atom1, atom2, atom3]
Available Next States: 50 states
Value Network Ranks All 50:
  [s3:0.9, s1:0.8, s7:0.75, ..., s50:0.1]
  
Top-20 Selected:
  [s3, s1, s7, s15, s23, ..., s42]
  
Policy Distribution:
  P(s3)=0.25, P(s1)=0.20, P(s7)=0.15, ... (over 20 actions)

═══════════════════════════════════════════════════════════

Time: t=500,000 (Mid-training)
Progress: 50%
K = 12 (linear schedule)

State: [atom4, atom5]
Available Next States: 50 states
Value Network Ranks All 50:
  [s12:0.95, s5:0.88, s33:0.82, ..., s8:0.05]
  
Top-12 Selected:
  [s12, s5, s33, s44, s2, ..., s19]
  
Policy Distribution:
  P(s12)=0.35, P(s5)=0.25, P(s33)=0.15, ... (over 12 actions)

═══════════════════════════════════════════════════════════

Time: t=1,000,000 (End of training)
Progress: 100%
K = 5 (final)

State: [atom6, atom7, atom8]
Available Next States: 50 states
Value Network Ranks All 50:
  [s18:0.98, s25:0.95, s3:0.91, s44:0.88, s7:0.85, ..., s2:0.02]
  
Top-5 Selected:
  [s18, s25, s3, s44, s7]
  
Policy Distribution:
  P(s18)=0.50, P(s25)=0.30, P(s3)=0.12, P(s44)=0.05, P(s7)=0.03

NOTE: More focused distribution, higher confidence
```

## Comparison: Fixed vs Curriculum

```
FIXED K=5 (Throughout Training)
════════════════════════════════════════════════════════════════

Time: t=0 (Start)
Progress: 0%
K = 5 (fixed)

Problem: Value network is untrained!
  Random/poor value estimates
  May prune good actions early
  Limited exploration

─────────────────────────────────────────────────────────────

CURRICULUM (20 → 5)
════════════════════════════════════════════════════════════════

Time: t=0 (Start)
Progress: 0%
K = 20 (broad exploration)

Benefit: 
  ✓ Value network can learn from diverse actions
  ✓ Discovers valuable state-action pairs
  ✓ Builds accurate value estimates

════════════════════════════════════════════════════════════════

Time: t=500,000 (Mid)
Progress: 50%
K = 12 (moderate focus)

Benefit:
  ✓ Value network is partially trained
  ✓ Can safely prune some low-value actions
  ✓ Balances exploration and exploitation

════════════════════════════════════════════════════════════════

Time: t=1,000,000 (End)
Progress: 100%
K = 5 (focused exploitation)

Benefit:
  ✓ Value network is well-trained
  ✓ Confidently selects best actions
  ✓ Fast, focused learning
  ✓ Same as Fixed K=5, but got here gradually
```

## Performance Over Time

```
Reward
  │
  │  ┌─────── Curriculum (20→5) ─────────┐  Higher peak
  │  │                                    │
1.0│  │         ┌──────────────────────────●  
  │  │        ╱                              
  │  │       ╱   
0.8│  │      ╱    Smoother learning
  │  │     ╱     
  │  │    ╱      
0.6│  │   ╱       
  │  │  ╱        
  │  │ ╱         
0.4│  │╱          
  │  ●           
  │  │   ┌─ Fixed K=5 ─┐
0.2│  │   │             │
  │  │   │     ╱───────●  Lower peak, more variance
  │  │   │    ╱
  │  │   │   ╱
  0├──┼───●──╱──────────────────────────────────────→ Time
     0   │      500k             1M
         └ Slow start (limited exploration)

Legend:
● Curriculum (20→5)
● Fixed K=5
```

## Action Space Visualization

```
Available Actions: 50 states
═══════════════════════════════════════════════════════════════

Early Training (K=20, 40% of actions):
┌─────────────────────────────────────────────────────────────┐
│ Top-20 Actions (Selectable)     │ Bottom-30 (Masked)        │
│ █████████████████████████████   │ ░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ Still broad exploration         │ Low-value actions pruned  │
└─────────────────────────────────────────────────────────────┘

Mid Training (K=12, 24% of actions):
┌─────────────────────────────────────────────────────────────┐
│ Top-12 Actions    │ Bottom-38 (Masked)                      │
│ ████████████████  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ Moderate focus    │ More aggressive pruning                 │
└─────────────────────────────────────────────────────────────┘

Late Training (K=5, 10% of actions):
┌─────────────────────────────────────────────────────────────┐
│ Top-5  │ Bottom-45 (Masked)                                 │
│ ██████ │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
│ Focused│ Aggressive pruning                                  │
└─────────────────────────────────────────────────────────────┘

Key: █ = Selectable actions, ░ = Masked out (logit = -∞)
```

## Integration Points

```
┌─────────────────────────────────────────────────────────────┐
│                    Full Training Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │ Data       │───→│ Environment  │───→│ Policy Network │ │
│  │ Handler    │    │              │    │                │ │
│  └────────────┘    └──────────────┘    └────────────────┘ │
│                           ↓                     ↓          │
│                    ┌──────────────┐    ┌────────────────┐ │
│                    │ Value        │←───│ Curriculum     │ │
│                    │ Network      │    │ Callback       │ │
│                    └──────────────┘    └────────────────┘ │
│                           ↓                     ↓          │
│                    Ranks Actions        Updates K          │
│                           ↓                     ↓          │
│                    ┌──────────────────────────────┐        │
│                    │ _filter_action_logits_top_k │        │
│                    │ Masks non-top-K actions     │        │
│                    └──────────────────────────────┘        │
│                                  ↓                         │
│                    ┌──────────────────────────────┐        │
│                    │ Sample action from top-K     │        │
│                    │ Execute in environment       │        │
│                    └──────────────────────────────┘        │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  Optional Integrations:                                      │
│  • KGE Integration (--kge_integration_strategy)             │
│  • Custom Rewards (--reward_type)                           │
│  • Multiple Environments (--n_envs)                         │
└─────────────────────────────────────────────────────────────┘
```

## Summary

The curriculum learning system provides a smooth transition from exploration to exploitation by dynamically adjusting the action space based on training progress. The value network ranks all actions, and only the top-K are presented to the policy, with K decreasing over time according to the chosen schedule.
