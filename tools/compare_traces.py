import argparse
import json
from math import isclose
from pathlib import Path


def load_trace(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def compare_traces(a, b, atol=1e-6, rtol=1e-6):
    n = min(len(a), len(b))
    for i in range(n):
        ra, rb = a[i], b[i]
        if ra.keys() != rb.keys():
            return False, i, f"keys differ: {ra.keys()} vs {rb.keys()}"
        for k in ra.keys():
            va, vb = ra[k], rb[k]
            if isinstance(va, float) and isinstance(vb, float):
                if not isclose(va, vb, abs_tol=atol, rel_tol=rtol):
                    return False, i, f"{k} float mismatch: {va} vs {vb}"
            else:
                if va != vb:
                    return False, i, f"{k} mismatch: {va} vs {vb}"
    if len(a) != len(b):
        return False, n, f"length mismatch: {len(a)} vs {len(b)}"
    return True, None, "traces match"


def main():
    parser = argparse.ArgumentParser(description="Compare two JSONL traces.")
    parser.add_argument("trace_a", type=Path)
    parser.add_argument("trace_b", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    trace_a = load_trace(args.trace_a)
    trace_b = load_trace(args.trace_b)
    ok, idx, msg = compare_traces(trace_a, trace_b, atol=args.atol, rtol=args.rtol)
    if ok:
        print("Traces match")
    else:
        print(f"Mismatch at record {idx}: {msg}")
        exit(1)


if __name__ == "__main__":
    main()
