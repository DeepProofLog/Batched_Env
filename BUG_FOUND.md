# Bug Found: Incorrect Success Determination in SB3 Environment

## Summary

The string-based environment (`str_based/str_env.py`) has a bug where proofs that succeed at exactly `max_depth` are incorrectly marked as failures.

## Root Cause

In `str_based/str_env.py` (lines 467-481):

```python
exceeded_max_depth = (self.current_depth >= self.max_depth)
...
done_next = done_next | exceeded_max_depth | truncate_flag
truncated = bool(exceeded_max_depth) or bool(truncate_flag)
...
if done_next:
    info["is_success"] = successful and not truncated
```

The problem: `truncated` is set to True whenever `max_depth` is reached, **regardless of whether the proof succeeded**. This means:
- If a proof succeeds at step 19 (depth < max_depth): `is_success = True`  
- If a proof succeeds at step 20 (depth == max_depth): `is_success = False` ❌

## Correct Behavior (BatchedEnv)

The tensor-based environment (`env.py`, lines 1075-1117) handles this correctly:

```python
depth_exceeded = self.current_depths >= self.max_depth
natural_term = all_true | any_false
terminated[:, 0] = natural_term
truncated[:, 0] = depth_exceeded & ~natural_term  # Only truncate if NOT naturally terminated
...
is_success[:, 0] = success  # success = all_true
```

This logic correctly distinguishes:
- **Natural termination** (proof completes): `terminated = True`, `is_success = all_true`
- **Truncation** (depth exceeded WITHOUT completing): `truncated = True`, `is_success = False`

## Evidence

Testing on countries_s3 dataset with deterministic policy:
- **SB3 env**: 73/159 successful (45.91%)
- **Tensor env**: 74/159 successful (46.54%)

Debug mode identified the first difference:
- **Query 0**: `locatedInCR(albania, europe)`
- **Both environments**: 21 steps, reward = 1.0
- **SB3 env**: `success = False` ❌
- **Tensor env**: `success = True` ✓

The proof completed at exactly step 20 (max_depth = 20). The SB3 env incorrectly marked it as truncated.

## Fix

Modify `str_based/str_env.py` line 471 to match the batched env logic:

```python
# OLD (incorrect):
truncated = bool(exceeded_max_depth) or bool(truncate_flag)

# NEW (correct):
natural_termination = bool(done_next) and not bool(exceeded_max_depth)
truncated = (bool(exceeded_max_depth) and not natural_termination) or bool(truncate_flag)
```

Or more simply, compute `successful` first and use it:

```python
truncated = (bool(exceeded_max_depth) and not successful) or bool(truncate_flag)
```

This ensures that if the proof succeeds (successful=True), it won't be marked as truncated even if max_depth was reached.

## Impact

This bug causes the SB3 environment to undercount successful proofs by marking some valid completions as failures. The magnitude depends on how many proofs complete at exactly max_depth.

For countries_s3 with max_depth=20: **1 out of 159 queries** (0.6%) affected.

## Test Results

After identifying this bug:
- All 3 **engines** match perfectly: 68/159 (42.77%) ✓
- **Environments** differ by 1 query due to this bug:
  - SB3 env: 73/159 (45.91%)
  - Tensor env: 74/159 (46.54%)
  - Batched tensor env: 74/159 (46.54%)

The difference between engines (42.77%) and environments (45-46%) is expected and by design - environments add terminal states for True/False atoms, slightly changing the proof search space.
