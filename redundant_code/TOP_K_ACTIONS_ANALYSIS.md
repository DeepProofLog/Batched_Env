# Top-K Action Selection Analysis

## Summary
**YES**, you have already implemented a mechanism to choose the top-k actions (e.g., top 5) according to the value function! The implementation is in `model.py` and is configurable via the `top_k_actions` parameter.

## Current Implementation

### Location
The implementation is in the `CustomActorCriticPolicy` class in `model.py`:

```python
def _filter_action_logits_top_k(
    self,
    action_logits: torch.Tensor,
    action_context: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Restrict logits to the top-k actions ranked by the value network."""
```

### How It Works

1. **Value-Based Ranking**: The method uses the **value network** to evaluate all available actions (states) and assigns each a value score.

2. **Top-K Selection**: It selects the K actions with the highest value scores using `torch.topk()`.

3. **Masking**: All actions outside the top-K are masked out by setting their logits to `-inf`, effectively preventing them from being selected by the policy.

4. **Respects Action Mask**: The filtering respects the original `action_mask`, ensuring only valid actions are considered.

### Key Implementation Details

```python
# Lines 420-462 in model.py

# Compute value for each action embedding
action_values = self.mlp_extractor.value_network(action_embeddings).detach()

# Mask invalid actions
action_values = action_values.masked_fill(~mask_bool, float('-inf'))

# Select top-k actions
k = min(self.top_k_actions, action_values.shape[1])
_, topk_indices = torch.topk(action_values, k=k, dim=1)

# Create selection mask
selection_mask = torch.zeros_like(action_values, dtype=torch.bool)
selection_mask.scatter_(1, topk_indices, True)
selection_mask &= mask_bool  # Combine with original mask

# Filter logits
filtered_logits = action_logits.masked_fill(~selection_mask, float('-inf'))
```

### Usage in Policy Methods

The filtering is applied in three critical policy methods:
- **`forward()`** (line 602): During action selection
- **`evaluate_actions()`** (line 641): During policy evaluation
- **`get_distribution()`** (line 680): When getting action distributions

## Configuration

### Setting Top-K Value

You can configure this in three ways:

#### 1. Command-line argument (recommended):
```bash
python runner.py --top_k_actions 5
```

#### 2. In `runner.py` config:
```python
TOP_K_ACTIONS = [5]  # Line 45
```

#### 3. In `train.py`:
```python
policy_kwargs = {
    'top_k_actions': 5,
    # ... other args
}
```

### Current Defaults
- **runner.py**: `TOP_K_ACTIONS = [5]` (line 45)
- **Disabled when**: `top_k_actions <= 0` or `top_k_actions = None`

## Advantages of Your Implementation

### ✅ Correct Design Choices

1. **Value-Based Selection**: Using the value network to rank actions is theoretically sound. The value function estimates the expected return from each state/action, making it an excellent heuristic for pruning.

2. **Proper Masking**: Setting non-selected actions to `-inf` in logit space is the correct way to exclude them from the probability distribution.

3. **Detached Values**: Using `.detach()` on value scores prevents gradients from flowing through the selection process, which is important for stable training.

4. **Dynamic K**: `k = min(self.top_k_actions, action_values.shape[1])` ensures it works even when there are fewer available actions than K.

5. **Fallback Handling**: When selection results in zero valid actions, it falls back to the original mask (lines 455-456).

6. **Integration**: The filtering is cleanly integrated into all three policy methods that need it.

## Theoretical Soundness

### Why This Works

In your case, **actions are also next states**, which makes this approach particularly appropriate:

1. **State-Action Equivalence**: Since each action leads to a specific next state, the value of an action is directly related to the value of the resulting state.

2. **Value Function as Heuristic**: The value function V(s) estimates the expected cumulative reward from state s. Using this to prioritize which next states to explore is a principled approach.

3. **Exploration-Exploitation Balance**: The agent still maintains a stochastic policy over the top-K actions, preserving some exploration while reducing the action space.

### Comparison to Alternatives

Your implementation is better than several alternatives:

❌ **Random Selection**: Would not leverage learned knowledge
❌ **Policy-Based Selection**: Would create circular dependency
✅ **Value-Based Selection**: Uses independent estimate of state quality
✅ **Q-Value Based**: Would be similar but require Q-network (you use V-network)

## Potential Considerations

### 1. Training Stability
- **Pro**: Reduces exploration space, potentially faster convergence
- **Con**: May prune valuable exploratory actions early in training
- **Recommendation**: Consider starting with higher K and gradually reducing it

### 2. Value Function Accuracy
- The effectiveness depends on how well your value network estimates state values
- Early in training, the value function may be inaccurate
- **Recommendation**: Monitor value function loss during training

### 3. Dynamic K Scheduling
You could implement adaptive K based on training progress:

```python
# Pseudo-code for adaptive K
if training_progress < 0.3:
    k = None  # No filtering early on
elif training_progress < 0.6:
    k = 10    # Moderate filtering
else:
    k = 5     # Aggressive filtering late
```

### 4. Alternative: Softmax Temperature
Instead of hard pruning, you could use temperature scaling:

```python
# Softer alternative (not implemented)
action_values = self.mlp_extractor.value_network(action_embeddings)
reweighted_logits = action_logits + temperature * action_values
```

## Testing

Your implementation includes comprehensive tests in `test_kge_integration.py`:

- ✅ `test_filter_action_logits_top_k_masks_to_highest_values()` (line 210)
- ✅ `test_filter_action_logits_top_k_no_truncation_when_k_large()` (line 258)
- ✅ `test_filter_action_logits_top_k_disabled_by_zero()` (line 283)

## Recommendations

### For Your Use Case (Actions = States)

1. **Current Implementation is Sound**: Your approach is theoretically correct and well-implemented.

2. **Recommended K Values**: 
   - Start with K=10 for initial training
   - Reduce to K=5 once value function is trained
   - Could go as low as K=3 for very large action spaces

3. **Monitor These Metrics**:
   - Value function loss
   - Entropy of action distribution (should decrease with lower K)
   - Success rate on validation set
   - Average number of valid actions per state

4. **Experiment Suggestions**:
   ```bash
   # Baseline (no filtering)
   python runner.py --top_k_actions 0
   
   # Conservative filtering
   python runner.py --top_k_actions 10
   
   # Moderate filtering (current default)
   python runner.py --top_k_actions 5
   
   # Aggressive filtering
   python runner.py --top_k_actions 3
   ```

5. **Consider Curriculum Learning**:
   - Train first epoch with `top_k_actions=None`
   - Switch to `top_k_actions=10` after value function stabilizes
   - Reduce to `top_k_actions=5` for final training

## Alternative Approaches (If Needed)

If you find the current approach insufficient, here are alternatives:

### 1. **Entropy-Regularized Filtering**
Only filter when entropy is above a threshold (to maintain exploration):

```python
if distribution.entropy().mean() > threshold:
    # Apply filtering
    action_logits = self._filter_action_logits_top_k(...)
```

### 2. **Q-Value Based Filtering**
If you had Q-values instead of state values:

```python
# Compute Q(s,a) for each action
q_values = self.q_network(state_embedding, action_embeddings)
_, topk_indices = torch.topk(q_values, k=k, dim=1)
```

### 3. **Hybrid Approach**
Combine value-based and policy-based scores:

```python
combined_scores = alpha * action_values + (1-alpha) * action_logits
_, topk_indices = torch.topk(combined_scores, k=k, dim=1)
```

### 4. **Learned Action Selection**
Train a separate network to predict which actions to keep:

```python
keep_probs = self.action_filter_network(state_embedding)
threshold = torch.topk(keep_probs, k=k, dim=1).values[:, -1:]
mask = keep_probs >= threshold
```

## Conclusion

**Your implementation is correct and well-designed.** The use of the value function to select top-K actions is theoretically sound, especially given that actions correspond to states in your domain.

**Key Strengths**:
- ✅ Proper integration with policy gradient methods
- ✅ Respects action masking
- ✅ Handles edge cases
- ✅ Comprehensive testing
- ✅ Configurable via command-line

**Suggested Next Steps**:
1. Run experiments with different K values (0, 3, 5, 10)
2. Monitor value function accuracy during training
3. Compare performance metrics across different K settings
4. Consider curriculum learning if needed

The implementation is production-ready and follows RL best practices!
