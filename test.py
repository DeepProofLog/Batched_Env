# test_depth5.py

import itertools # Make sure to import if resetting counters
from utils import Term, Rule
from depth_python_prover_v2 import calculate_provability_ratios_by_depth # Adjust import path if needed

# --- Test Case: Depth 5 Required ---

# Facts
facts_db_depth5 = [
    Term("base_property", ["item1"]),
    Term("relation", ["item1", "attribute_a"]),
    Term("config", ["attribute_a", "setting_alpha"]),
]

# Rules
# target(X) needs goal4(X, setting_alpha)
# goal4(A, C) needs goal3(A, B) and config(B, C)
# goal3(Y, Z) needs goal2(Y) and relation(Y, Z)
# goal2(P) needs goal1(P, item1)
# goal1(Q, R) needs base_property(R)  <- Note: This links Q to R via variable, R must match base_property
#                                       If query is target(item1), then Q=item1, R=item1. Needs base_property(item1)

rules_db_depth5 = [
    # Rule 1 (Leads to Depth 5)
    Rule(Term("target", ["X"]),
         [Term("goal4", ["X", "setting_alpha"])]),

    # Rule 2 (Leads to Depth 4) - Needs goal3(A, B) AND config(B, C)
    Rule(Term("goal4", ["A", "C"]),
         [Term("goal3", ["A", "B"]), Term("config", ["B", "C"])]),

    # Rule 3 (Leads to Depth 3) - Needs goal2(Y) AND relation(Y, Z)
    Rule(Term("goal3", ["Y", "Z"]),
         [Term("goal2", ["Y"]), Term("relation", ["Y", "Z"])]),

    # Rule 4 (Leads to Depth 2)
    Rule(Term("goal2", ["P"]),
         [Term("goal1", ["P", "item1"])]), # Hardcodes 'item1' for the second arg

    # Rule 5 (Leads to Depth 1)
    Rule(Term("goal1", ["Q", "R"]),       # Q and R are variables here
         [Term("base_property", ["R"])])   # Body only depends on R
]

# Query
query_depth5 = Term("target", ["item1"]) # We expect X=item1 throughout

# Expected Minimum Rule Depth: 5
# Expected Resolution Depth (from your prover): 6

# --- Run the Test ---
if __name__ == "__main__":
    # Optional: Reset standardization counters for consistent test runs
    # try:
    #     from prolog_like_unification import _standardize_counter, _state_var_counter
    #     _standardize_counter = itertools.count()
    #     _state_var_counter = itertools.count()
    # except ImportError:
    #     print("Warning: Could not reset counters.")
    #     pass

    print("\n--- Running Depth 5 Test Case ---")
    max_depth_check = 10 # Check up to depth 10 to be safe

    provability_ratios_d5, proved_queries_d5 = calculate_provability_ratios_by_depth(
        queries=[query_depth5], # Pass the single query
        max_depth_check=max_depth_check,
        rules=rules_db_depth5,
        facts=facts_db_depth5,
        is_train_data=False, # Assuming not filtering facts
    )

    # Print results specific to this test
    print("\n--- Depth 5 Test Results ---")
    expected_rule_depth = 5
    expected_resolution_depth = 6 # Based on previous analysis of your code

    if proved_queries_d5:
        result_depth = proved_queries_d5.get(str(query_depth5))
        if result_depth is not None:
            print(f"Query '{query_depth5}' proven at depth: {result_depth}")
            if result_depth == expected_resolution_depth:
                print(f"Resolution depth calculation CORRECT ({result_depth}). Expected Rule depth was {expected_rule_depth}.")
            elif result_depth == expected_rule_depth:
                 print(f"Depth calculation ({result_depth}) matches RULE depth, not expected RESOLUTION depth ({expected_resolution_depth}). Check prover logic/definition.")
            else:
                print(f"Depth calculation INCORRECT. Expected Rule Depth {expected_rule_depth}, Expected Resolution Depth {expected_resolution_depth}, Got {result_depth}.")
        else:
            print(f"Query '{query_depth5}' reported provable but depth not found in results dict.")
    else:
        # Check ratios dict if proof wasn't recorded correctly
        final_ratio = provability_ratios_d5.get(max_depth_check, 0.0)
        print(f"Provability Ratio at depth {max_depth_check}: {final_ratio}")
        if final_ratio > 0.0:
             print(f"Query '{query_depth5}' was likely proven but not recorded correctly in proved_queries.")
        else:
             print(f"Query '{query_depth5}' was NOT proven within depth {max_depth_check}.")

    # Print the ratios_by_depth dictionary for detailed view
    print("Provability Ratios:", provability_ratios_d5)