from typing import Dict, List, Tuple


def print_traces(trace1: Dict, trace2: Dict) -> None:
    """
    DEBUG MODE: print both traces side-by-side for comparison.
    """
    mssg_lines = []
    return mssg_lines




def compare_traces_dicts(trace1: List[Dict], trace2: List[Dict]) -> bool:
    """
    Compare two trace dicts step-by-step and point out differences.
    
    Args:
        trace1: First trace dict, expected to have 'state' and 'steps'
        trace2: Second trace dict, expected to have 'state' and 'steps'
    
    Returns:
        True if traces match, False otherwise (and prints differences)
    """
    for i, (step1, step2) in enumerate(zip(trace1, trace2)):
        for key in step1:
            if key not in step2:
                print(f"Key '{key}' missing in second trace")
                mss = print_traces(trace1, trace2)
                print(mss)
                return False
            elif step1[key] != step2[key]:
                print(f"Difference in key '{key}':")
                print(f"  Trace1: {step1[key]}")
                print(f"  Trace2: {step2[key]}")
                mss = print_traces(trace1, trace2)
                print(mss)
                return False
        for key in step2:
            if key not in step1:
                print(f"Key '{key}' missing in first trace")
                mss = print_traces(trace1, trace2)
                print(mss)
                return False
    return True