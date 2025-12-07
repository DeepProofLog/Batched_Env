
import os
import re
import sys
import pytest
import subprocess
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = ROOT / "runner.py"
SB3_RUNNER_SCRIPT = ROOT / "sb3" / "sb3_runner.py"

def parse_mrr(output: str) -> float:
    """Parse MRR from output."""
    print(f"DEBUG: Parsing output length {len(output)}")
    
    # 1. Look for "mrr_mean: X" (Tensor/SB3 final stats) - PREFERRED
    # Search from end (last occurrence) to get final result
    # Allow optional spaces around colon
    matches = re.findall(r"mrr_mean\s*:\s+([\d\.]+)", output)
    if matches:
        print(f"DEBUG: Found 'mrr_mean' (last): {matches[-1]}")
        return float(matches[-1]) 

    # 2. Look for explicit "New best MRR in eval: X" (SB3 common) as fallback
    # This might be pre-update, so less reliable for final parity
    match = re.search(r"New best MRR in eval: ([\d\.]+)", output)
    if match:
        print(f"DEBUG: Found 'New best MRR': {match.group(1)}")
        return float(match.group(1))

    # 3. Look for "SB3 MRR: X" or "Tensor MRR: X" if printed by test helper
    match = re.search(r"(?:SB3|Tensor)\s+MRR:\s+([\d\.]+)", output)
    if match:
         print(f"DEBUG: Found 'Runner MRR': {match.group(1)}")
         return float(match.group(1))

    print(f"DEBUG: No MRR found.")
    return 0.0

def parse_mrr_from_csv(script_cmd: list) -> float:
    """Find the latest CSV and parse MRR from filename."""
    # Run signature inferred from args
    # e.g. countries_s3-...
    # Just look for latest csv in runs/indiv_runs/
    import glob
    indiv_runs = ROOT / "runs" / "indiv_runs"
    if not indiv_runs.exists():
        print(f"DEBUG: {indiv_runs} does not exist.")
        return 0.0
        
    # Get latest file
    files = list(indiv_runs.glob("*.csv"))
    if not files:
        print("DEBUG: No CSV files found.")
        return 0.0
    
    latest_file = max(files, key=os.path.getctime)
    print(f"DEBUG: Found latest CSV: {latest_file.name}")
    
    # Parse MRR from filename: ..._{reward}_{mrr}-seed...
    # e.g. ...-0.875_0.889-seed_0.csv
    try:
        parts = latest_file.name.split('-')
        # Find part with underscore matching float_float
        for part in parts:
            if '_' in part and part.replace('_', '').replace('.', '').isdigit():
                metrics = part.split('_')
                if len(metrics) == 2:
                    mrr = float(metrics[1])
                    print(f"DEBUG: Parsed MRR from filename: {mrr}")
                    return mrr
    except Exception as e:
        print(f"DEBUG: Failed to parse filename: {e}")
        
    return 0.0

def run_script(script_path: Path, args: list) -> tuple[float, str]:
    """Run a script and return (mrr, full_output)."""
    cmd = [sys.executable, str(script_path)] + args
    
    # Force parity env vars
    env = os.environ.copy()
    env['USE_FAST_CATEGORICAL'] = '0'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['PYTHONHASHSEED'] = '0'
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr) # Print stderr for debug
        return 0.0, result.stdout + "\n" + result.stderr

    # Try parsing stdout first
    mrr = parse_mrr(result.stdout)
    
    # If stdout parsing failed (0.0), try CSV
    if mrr == 0.0:
        print("DEBUG: Stdout parsing failed, checking CSVs...")
        mrr = parse_mrr_from_csv(cmd)
        
    return mrr, result.stdout

def test_script_parity():
    """Run both scripts and compare results."""
    # Settings for fast deterministic run
    # batch_size=800 matches n_steps*n_envs (40*20)
    # timesteps_train=0 verifies INITIALIZATION parity
    common_args = [
        "--set", "timesteps_train=0",
        "--set", "dataset_name=countries_s3",
        "--set", "seed=[0]",
        "--set", "batch_size=800"  
    ]
    
    print("\n>>> Running SB3 Runner...")
    sb3_mrr, sb3_out = run_script(SB3_RUNNER_SCRIPT, common_args)
    print(f"SB3 MRR: {sb3_mrr}")
    
    print("\n>>> Running Tensor Runner...")
    # Tensor needs to disable AMP and Compile to match SB3 float32 exactly
    tensor_args = common_args + ["--set", "use_amp=False", "--set", "use_compile=False"]
    tensor_mrr, tensor_out = run_script(RUNNER_SCRIPT, tensor_args)
    print(f"Tensor MRR: {tensor_mrr}")
    
    # Debug output if mismatch
    if abs(sb3_mrr - tensor_mrr) > 0.001:
        print("!!! MISMATCH DETECTED !!!")
        print("-" * 20 + " SB3 Output Tail " + "-" * 20)
        print(sb3_out[-2000:])
        print("-" * 20 + " Tensor Output Tail " + "-" * 20)
        print(tensor_out[-2000:])
    
    assert sb3_mrr > 0, "SB3 run failed to produce MRR"
    assert tensor_mrr > 0, "Tensor run failed to produce MRR"
    assert abs(sb3_mrr - tensor_mrr) < 0.001, f"Mismatch: SB3={sb3_mrr}, Tensor={tensor_mrr}"

if __name__ == "__main__":
    test_script_parity()
    print("\nSUCCESS: Scripts produce identical results!")
