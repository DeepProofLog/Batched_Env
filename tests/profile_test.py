"""
Profile the batched environment to find bottlenecks
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import cProfile
import pstats
import io
from tests.test_rollout import test_rollout_pipeline as test_pipeline

def profile_test():
    """Profile the test with cProfile"""
    # use cuda if available, but for profiling we might want CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run test with smaller batch

    test_pipeline()

    profiler.disable()
    
    # Print profiling results
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top Time-Consuming Functions")
    print("="*80)
    n_functions = 15
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top by Total Time")
    print("="*80)
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('tottime')
    ps.print_stats(n_functions)
    print(s.getvalue())
    
    # save profiling results from print stats in cumulative and tottime to file
    with open('tests/profile_results.txt', 'w') as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        ps.print_stats(n_functions)
        f.write("\n\n")
        ps.sort_stats('tottime')
        ps.print_stats(n_functions)


if __name__ == '__main__':
    profile_test()