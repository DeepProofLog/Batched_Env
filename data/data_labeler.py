import os
import janus_swi as janus
import re
import argparse
import json
import ast

def get_prolog_rules(file_dir):
    with open(file_dir, "r") as f:
        predicates = {}
        rules = []
        for line in f:
            rule = line.strip().split(":")[-1]
            body = rule.split("->")[0].strip()
            head = rule.split("->")[1].strip()
            prolog_rule = f"{head} :- {body}.\n"
            rules.append(prolog_rule)
            matches = re.findall(r'(\b\w+)\(([^)]*)\)', prolog_rule)
            predicates_with_arity = [(predicate, len(args.split(','))) for predicate, args in matches]
            for predicate, arity in predicates_with_arity:
                if predicate not in predicates:
                    predicates[predicate] = arity
                # surpass_warnings.append(f":- discontiguous {predicate}/{arity}.\n")
    return predicates, rules

def get_prolog_facts(file_dirs):
    facts = []
    predicates = {}
    for file in file_dirs:
        if os.path.exists(file):
            with open(file, "r") as f:
                facts.extend(f.readlines())
    for f in facts:
        matches = re.findall(r'(\b\w+)\(([^)]*)\)', f)
        predicates_with_arity = [(predicate, len(args.split(','))) for predicate, args in matches]
        for predicate, arity in predicates_with_arity:
            if predicate not in predicates:
                predicates[predicate] = arity
    return predicates, facts

def get_pl(rule_file, fact_files, output_file, catch_errors, use_tabling):
    predicates_1, rules = get_prolog_rules(rule_file)
    predicates_2, facts = get_prolog_facts(fact_files)
    predicates = {**predicates_1, **predicates_2}
    outputs = []
    full_facts = []
    with open(output_file, "w") as f:
        for predicate, arity in predicates.items():
            outputs.append(f":- discontiguous {predicate}/{arity}.\n")
            full_facts.append(f":- discontiguous {predicate}/{arity}.\n")
            f.write(f":- discontiguous {predicate}/{arity}.\n")
        if use_tabling:
            outputs.append(":- table locatedInCR/2.\n")
            full_facts.append(":- table locatedInCR/2.\n")
            f.write(":- table locatedInCR/2.\n")
        for fact in facts:
            outputs.append(fact)
            full_facts.append(fact)
            f.write(fact)
        if catch_errors:
            outputs.append("call_with_catch(Goal, TimeOut) :- catch((call_with_time_limit(60, Goal), TimeOut=false), time_limit_exceeded, (writeln('Query timed out'), TimeOut=true)).\n")
            f.write("call_with_catch(Goal, TimeOut) :- catch((call_with_time_limit(60, Goal), TimeOut=false), time_limit_exceeded, (writeln('Query timed out'), TimeOut=true)). \n")
        for rule in rules:
            outputs.append(rule)
            f.write(rule)
    return "".join(outputs), "".join(full_facts)


def get_labeled_data(query_file, catch_errors, use_modified_rules, full_rule):
    with open(query_file, "r") as f:
        queries = f.readlines()
    outputs = []
    for query in queries:
        query = query.strip()
        if query.startswith("locatedInCR"):
            #print(query)
            full_rules = full_rule.split("\n")
            #print(len(full_rules))
            if query in full_rules:
                #print("query in full_rules")
                full_rules.remove(query)
            with open("countries_sub.pl", "w") as f1:
                #print(len(full_rules))
                f1.write("\n".join(full_rules))
            f1.close()
            #print("\n".join(full_rules))
            janus.query_once("abolish(neighborOf/2).")
            janus.query_once("abolish(locatedInCR/2).")
            janus.query_once("abolish_all_tables.")
            janus.consult("countries_sub.pl")
            if catch_errors:
                try:
                    res = janus.query_once(f"call_with_catch({query[:-1]}, TimeOut)")
                    #for d in res:
                    if res["TimeOut"] == "true":
                        output = f"{query}\ttimeout\n"
                    else:
                        output = f"{query}\t{res['truth']}\n"
                except janus.PrologError as e:
                    print(e)
                    if "Stack limit (1.0Gb) exceeded" in str(e):
                        output = f"{query}\tstack_limit_exceeded\n"
                print(output)
                outputs.append(output)
            else:
                # res = janus.query_once("listing.")
                # print(res)
                res = janus.query_once(query)
                #for d in res:
                output = f"{query}\t{res['truth']}\n"
                print(output)
                outputs.append(output)
    if not use_modified_rules:
        output_dir = query_file.split(".")[0] + "_label.txt"
    else:
        output_dir = query_file.split(".")[0] + "_mod_label.txt"
    with open(output_dir, "w") as f:
        for output in outputs:
            f.write(output)
    return None


def get_one_depth_proof(query, rules, full_facts, max_depth):
    #print(query)
    facts = full_facts.split("\n")
    if query in facts:
        #print("remove query")
        facts.remove(query)
    full_rules = facts + rules
    with open("countries_sub.pl", "w") as f1:
        f1.write("\n".join(full_rules))
    f1.close()
    janus.query_once("abolish(neighborOf/2).")
    janus.query_once("abolish(locatedInCR/2).")
    janus.query_once("abolish_all_tables.")
    janus.consult("countries_sub.pl")
    args = query.split("(")[1].replace(").", "").split(",")
    min_depth = 100
    min_proof = None
    # res1 = janus.query_once(
    #     f'locatedInCR_with_depth({args[0]}, {args[1]}, 0, {max_depth}, _Depths, _Proofs), term_string(_Depths, Depths), term_string(_Proofs, Proofs)')
    # print(res1)
    res = janus.query(f'locatedInCR_with_depth({args[0]}, {args[1]}, 0, {max_depth}, _Depths, _Proofs), term_string(_Depths, Depths), term_string(_Proofs, Proofs)')
    # print(res["Depths"])
    for r in res:
        print(r["Depths"])
        if int(r["Depths"]) < min_depth:
            min_depth = int(r["Depths"])
            min_proof = r["Proofs"]
        if min_depth == 2:
            break
    if not min_proof == None:
        return min_depth, min_proof
    else:
        return None, None

def get_depth_proof(inf_dir, rule_dir, full_facts, max_depth):
    with open(rule_dir, "r") as f:
        rules = f.readlines()
    # print(rules)

    with open(inf_dir, "r") as f:
        queries = json.load(f)
    outputs = []
    for query, corruptions in queries.items():
        min_depth, min_proof = get_one_depth_proof(query, rules, full_facts, max_depth)
        value = "True"
        #print(query, value, min_depth, min_proof)
        outputs.append([query, str(value), str(min_depth), str(min_proof)])
        for corruption in corruptions:
            q = corruption[0]
            value = corruption[1]
            min_depth, min_proof = get_one_depth_proof(q, rules, full_facts, max_depth)
            # print(q, value, min_depth, min_proof)
            outputs.append([q, str(value), str(min_depth), str(min_proof)])
    with open(inf_dir.split(".")[0] + "_depth_proof.txt", "w") as f:
        for output in outputs:
            print(output)
            f.write("\t".join(output) + "\n")
    return None



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--folder', default='countries', type=str)
    arg_parser.add_argument('--level', default='s3', type=str)
    arg_parser.add_argument('--catch_errors', action='store_true')
    arg_parser.add_argument('--use_modified_rules', action='store_true')
    arg_parser.add_argument('--use_tabling', action='store_false')
    args = arg_parser.parse_args()

    root_dir = f"{args.folder}_{args.level}/"
    if args.use_modified_rules:
        full_rules, full_facts = get_pl(root_dir + "rules_mod.txt",[root_dir + "facts.txt", root_dir + "train.txt"], root_dir + "countries_mod.pl",
              args.catch_errors, args.use_tabling)
        #janus.consult(root_dir+"countries_mod.pl")
    else:
        full_rules, full_facts = get_pl(root_dir+"rules.txt", [root_dir+"facts.txt", root_dir+"train.txt"], root_dir+"countries.pl", args.catch_errors, args.use_tabling)
        #janus.consult(root_dir+"countries.pl")

    print("processing train.txt")
    get_labeled_data(root_dir+"train.txt", args.catch_errors, args.use_modified_rules, full_rules)
    print("processing valid.txt")
    get_labeled_data(root_dir+"valid.txt", args.catch_errors, args.use_modified_rules, full_rules)
    print("processing test.txt")
    get_labeled_data(root_dir+"test.txt", args.catch_errors, args.use_modified_rules, full_rules)

    # max_depth = 10
    # print("processing train_label_corruptions.json")
    # get_depth_proof(root_dir + "train_label_corruptions.json", root_dir + "rules_with_depth.txt", full_facts, max_depth)
    # print("processing valid_label_corruptions.json")
    # get_depth_proof(root_dir+"valid_label_corruptions.json", root_dir + "rules_with_depth.txt", full_facts, max_depth)
    # print("processing test_label_corruptions.json")
    # get_depth_proof(root_dir+"test_label_corruptions.json", root_dir + "rules_with_depth.txt", full_facts, max_depth)