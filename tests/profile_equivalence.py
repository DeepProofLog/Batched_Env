"""Profile the test_engine_equivalence to find performance bottlenecks"""
import cProfile
import pstats
import io
from pstats import SortKey
import time

# Run the test for a small number of queries
import sys
sys.argv = ['test_engine_equivalence.py']  # Reset args to avoid parsing issues

# include root path
import os
import sys
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def run_test():
    import torch
    from data_handler import DataHandler
    from str_dataset import DataHandler as StrDataHandler
    from index_manager import IndexManager
    from str_index_manager import IndexManager as StrIndexManager
    from unification_engine import UnificationEngine
    from debug_helper import DebugHelper
    from str_utils import Term as StrTerm
    from str_unification import get_next_unification_python
    from test_engine_equivalence import test_single_query, canonicalize_str_state, canonicalize_tensor_state
    
    # Load data
    dh_str = StrDataHandler(
        dataset_name='family',
        base_path='./data/',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt',
        train_depth=None,
    )
    
    dh_non = DataHandler(
        dataset_name='family',
        base_path='./data/',
        train_file='train.txt',
        valid_file='valid.txt',
        test_file='test.txt',
        rules_file='rules.txt',
        facts_file='train.txt',
        train_depth=None,
    )
    
    # Setup string engine
    im_str = StrIndexManager(
        constants=dh_str.constants,
        predicates=dh_str.predicates,
        max_total_vars=1000000,
        rules=dh_str.rules,
        padding_atoms=20,
        max_arity=dh_str.max_arity,
        device=None,
    )
    fact_index_str = im_str.build_fact_index(dh_str.facts)
    rules_by_pred = {}
    for r in dh_str.rules:
        rules_by_pred.setdefault(r.head.predicate, []).append(r)
    facts_set_str = frozenset(dh_str.facts)
    
    # Setup tensor engine
    im_non = IndexManager(
        constants=dh_non.constants,
        predicates=dh_non.predicates,
        max_total_runtime_vars=1000000,
        padding_atoms=20,
        max_arity=dh_non.max_arity,
        device=torch.device('cpu'),
        rules=dh_non.rules,
    )
    dh_non.materialize_indices(im=im_non, device=torch.device('cpu'))
    
    stringifier_params = {
        'verbose': 0,
        'idx2predicate': im_non.idx2predicate,
        'idx2constant': im_non.idx2constant,
        'idx2template_var': im_non.idx2template_var,
        'padding_idx': im_non.padding_idx,
        'n_constants': im_non.constant_no
    }
    
    engine = UnificationEngine.from_index_manager(im_non, take_ownership=True, stringifier_params=stringifier_params)
    debug_helper = DebugHelper(
        verbose=0,
        idx2predicate=im_non.idx2predicate,
        idx2constant=im_non.idx2constant,
        idx2template_var=im_non.idx2template_var,
        padding_idx=engine.padding_idx,
        n_constants=im_non.constant_no
    )
    
    # Prepare data structures
    str_engine_data = (dh_str, im_str, fact_index_str, rules_by_pred)
    max_template_var = engine.constant_no
    if engine.rules_idx.numel() > 0:
        rule_max = engine.rules_idx.max().item()
        if rule_max > max_template_var:
            max_template_var = rule_max
    next_var_start = max_template_var + 1
    tensor_engine_data = (dh_non, im_non, engine, debug_helper, next_var_start)
    
    # Test first 20 queries
    print("Testing 20 queries...")
    start = time.time()
    for i in range(20):
        q = dh_str.train_queries[i]
        p, h, t = q.predicate, q.args[0], q.args[1]
        
        try:
            match, str_success, tensor_success, reason = test_single_query(
                p, h, t,  str_engine_data, tensor_engine_data, split='train', verbose=False
            )
            elapsed = time.time() - start
            print(f"Query {i}: {p}({h},{t}) - {'✓' if match else 'X'} ({elapsed:.2f}s total)")
        except Exception as e:
            print(f"Query {i}: ERROR - {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            break

# Profile the run
pr = cProfile.Profile()
pr.enable()

run_test()

pr.disable()

# Print stats
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
ps.print_stats(30)  # Top 30 functions
print(s.getvalue())
