"""
Profile the batched environment to find bottlenecks
"""
import torch
import cProfile
import pstats
import io
from test_batched_env import test_batched_env

def profile_test():
    """Profile the test with cProfile"""
    pr = cProfile.Profile()
    pr.enable()
    
    # Run test with smaller batch
    test_batched_env(
        dataset_name='wn18rr',
        batch_size=16,
        num_rollouts=3,
        max_steps_per_rollout=5,
        verbose=0
    )
    
    pr.disable()
    
    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())

if __name__ == '__main__':
    profile_test()
