#!/usr/bin/env python3
"""
Compare batched and SB3 trace files to verify exact matching.
"""
import json
import sys
from collections import defaultdict

def load_trace(filename):
    """Load trace file and organize by phase/iteration/step/env"""
    traces = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    with open(filename) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            phase = entry.get("phase", "unknown")
            iteration = entry.get("iteration", 0)
            step = entry.get("step", 0)
            env = entry.get("env", 0)
            traces[phase][iteration][step][env] = entry
    return traces

def compare_values(key, val1, val2, tolerance=1e-6):
    """Compare two values with tolerance for floats"""
    import math
    if isinstance(val1, (int, bool)) and isinstance(val2, (int, bool)):
        return val1 == val2, None
    elif isinstance(val1, float) and isinstance(val2, float):
        # Handle special float values
        if math.isnan(val1) and math.isnan(val2):
            return True, None  # Both NaN
        if math.isinf(val1) and math.isinf(val2):
            # Check if both have same sign of infinity
            if (val1 > 0) == (val2 > 0):
                return True, None
            else:
                return False, f"{val1} != {val2}"
        if math.isnan(val1) or math.isnan(val2) or math.isinf(val1) or math.isinf(val2):
            # One is special, the other isn't
            return False, f"{val1} != {val2}"
        # Normal float comparison
        diff = abs(val1 - val2)
        match = diff <= tolerance
        return match, diff if not match else None
    elif isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False, f"Length mismatch: {len(val1)} vs {len(val2)}"
        for i, (v1, v2) in enumerate(zip(val1, val2)):
            match, err = compare_values(f"{key}[{i}]", v1, v2, tolerance)
            if not match:
                return False, f"Index {i}: {err}"
        return True, None
    elif val1 == val2:
        return True, None
    else:
        return False, f"{val1} != {val2}"

def compare_traces(batched_file, sb3_file, max_mismatches=10):
    """Compare two trace files and report differences"""
    print(f"Loading {batched_file}...")
    batched = load_trace(batched_file)
    print(f"Loading {sb3_file}...")
    sb3 = load_trace(sb3_file)
    
    all_phases = set(batched.keys()) | set(sb3.keys())
    total_mismatches = 0
    total_comparisons = 0
    
    for phase in sorted(all_phases):
        print(f"\n{'='*80}")
        print(f"Phase: {phase}")
        print(f"{'='*80}")
        
        if phase not in batched:
            print(f"  ❌ Phase {phase} missing in batched traces")
            continue
        if phase not in sb3:
            print(f"  ❌ Phase {phase} missing in SB3 traces")
            continue
        
        batched_phase = batched[phase]
        sb3_phase = sb3[phase]
        
        all_iterations = set(batched_phase.keys()) | set(sb3_phase.keys())
        
        for iteration in sorted(all_iterations):
            if iteration not in batched_phase:
                print(f"  ❌ Iteration {iteration} missing in batched traces")
                continue
            if iteration not in sb3_phase:
                print(f"  ❌ Iteration {iteration} missing in SB3 traces")
                continue
            
            batched_iter = batched_phase[iteration]
            sb3_iter = sb3[phase][iteration]
            
            all_steps = set(batched_iter.keys()) | set(sb3_iter.keys())
            
            for step in sorted(all_steps):
                if step not in batched_iter:
                    print(f"  ❌ Step {step} missing in batched traces")
                    total_mismatches += 1
                    continue
                if step not in sb3_iter:
                    print(f"  ❌ Step {step} missing in SB3 traces")
                    total_mismatches += 1
                    continue
                
                batched_step = batched_iter[step]
                sb3_step = sb3_iter[step]
                
                all_envs = set(batched_step.keys()) | set(sb3_step.keys())
                
                for env in sorted(all_envs):
                    if env not in batched_step:
                        print(f"  ❌ Env {env} at step {step} missing in batched traces")
                        total_mismatches += 1
                        continue
                    if env not in sb3_step:
                        print(f"  ❌ Env {env} at step {step} missing in SB3 traces")
                        total_mismatches += 1
                        continue
                    
                    batched_entry = batched_step[env]
                    sb3_entry = sb3_step[env]
                    total_comparisons += 1
                    
                    # Compare key fields
                    fields_to_compare = [
                        "action", "reward", "done", "length", "value", "log_prob",
                        "sub_index", "derived_sub_indices", "action_mask", "logits"
                    ]
                    
                    step_has_mismatch = False
                    for field in fields_to_compare:
                        if field not in batched_entry and field not in sb3_entry:
                            continue
                        if field not in batched_entry:
                            print(f"  ❌ [{phase}] iter={iteration} step={step} env={env}: field '{field}' missing in batched")
                            step_has_mismatch = True
                            continue
                        if field not in sb3_entry:
                            print(f"  ❌ [{phase}] iter={iteration} step={step} env={env}: field '{field}' missing in SB3")
                            step_has_mismatch = True
                            continue
                        
                        batched_val = batched_entry[field]
                        sb3_val = sb3_entry[field]
                        
                        match, error = compare_values(field, batched_val, sb3_val)
                        if not match:
                            print(f"  ❌ [{phase}] iter={iteration} step={step} env={env}: {field} mismatch")
                            print(f"     Batched: {batched_val}")
                            print(f"     SB3:     {sb3_val}")
                            if error:
                                print(f"     Error:   {error}")
                            step_has_mismatch = True
                            total_mismatches += 1
                            
                            if total_mismatches >= max_mismatches:
                                print(f"\n⚠️  Reached maximum of {max_mismatches} mismatches, stopping comparison")
                                print(f"✅ Total comparisons: {total_comparisons}")
                                print(f"❌ Total mismatches: {total_mismatches}")
                                return total_mismatches
                    
                    if not step_has_mismatch and (step == 0 or step % 100 == 0):
                        print(f"  ✅ [{phase}] iter={iteration} step={step} all {len(all_envs)} envs match")
    
    print(f"\n{'='*80}")
    print(f"Comparison Summary")
    print(f"{'='*80}")
    print(f"✅ Total comparisons: {total_comparisons}")
    print(f"❌ Total mismatches: {total_mismatches}")
    
    if total_mismatches == 0:
        print(f"\n🎉 All traces match exactly!")
    else:
        print(f"\n⚠️  Found {total_mismatches} mismatches")
    
    return total_mismatches

if __name__ == "__main__":
    batched_file = "traces_comparison/batched_trace.jsonl"
    sb3_file = "traces_comparison/sb3_trace.jsonl"
    
    if len(sys.argv) > 1:
        batched_file = sys.argv[1]
    if len(sys.argv) > 2:
        sb3_file = sys.argv[2]
    
    max_mismatches = 50  # Stop after finding this many mismatches
    if len(sys.argv) > 3:
        max_mismatches = int(sys.argv[3])
    
    mismatches = compare_traces(batched_file, sb3_file, max_mismatches)
    sys.exit(0 if mismatches == 0 else 1)
