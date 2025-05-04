'''
Get test files for all possible depths, including the covered queries by the rules

1. Use the depth_prolog_prover to get the depth of the queries (set a max depth). 
It generates data/dataset/{set}_depths.txt files for train/val/test.

2. For every depth, write the queries to a file named test_d<depth>.txt
 (inclues also the non proved queries and the queries covered by the rules).
'''
import os
import re
from get_pl import get_prolog_rules

dataset = 'wn18rr'

root = os.getcwd()
root_dir = f"{root}/data/{dataset}/"
rules_file = f"{root_dir}rules.txt"

for set in ["train", "valid", "test"]:
    set_file = f"{root_dir}{set}_depths.txt"

    # 1. write the test files for each depth

    # Read the queries and their depths (int or -1)
    queries = {}
    print(f"Reading {set_file}...")
    with open(set_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            query = line.strip().split(" ")[0]
            query_depth = line.strip().split(" ")[1]
            query_depth = int(query_depth) if query_depth != "-1" else -1
            queries[query] = query_depth


    # 2. For every depth different from -1, write the queries to a file named test_d<depth>.txt
    max_depth = max([depth for depth in queries.values() if depth != -1])
    for depth in range(max_depth + 1):
        if any(d == depth for d in queries.values()):
            folder = f"{root_dir}/depths/{set}_d{depth}.txt"
            if not os.path.exists(os.path.dirname(folder)):
                os.makedirs(os.path.dirname(folder))
            with open(folder, "w") as f:
                for query, d in queries.items():
                    if d == depth:
                        f.write(f"{query}\n")

    # For the non proved queries, write them to a file named {set}_nonproved.txt

    non_proved_file = f"{root_dir}/depths/{set}_nonproved.txt"
    with open(non_proved_file, "w") as f:
        for query, d in queries.items():
            if d == -1:
                f.write(f"{query}\n")

    # 3. Read the rules and get the predicates
    predicates, rules = get_prolog_rules(rules_file)

    # get the queries whose predicates are in the rules
    count = 0
    covered_queries = []
    for query, d in queries.items():
        count += 1
        predicates_query = re.findall(r"\w+", query)
        predicates_query = query.split("(")[0]
        print(f"Query: {query}, Depth: {d}, Predicates: {predicates_query}") if count < 30 else None
        # check if any of the predicates are in the rules
        for predicate in predicates_query:
            if predicate in predicates:
                covered_queries.append(query)
                break
            
    # write the covered queries to a file named test_covered.txt
    covered_file = f"{root_dir}/depths/{set}_covered.txt"
    if not os.path.exists(os.path.dirname(covered_file)):
        os.makedirs(os.path.dirname(covered_file))
    with open(covered_file, "w") as f:
        for query in covered_queries:
            f.write(f"{query}\n")



