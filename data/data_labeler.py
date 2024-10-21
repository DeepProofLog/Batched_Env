import os
import janus_swi as janus
import re
import argparse

from jinja2.nodes import args_as_const


def get_prolog_rules(file_dir):
    with open(file_dir, "r") as f:
        surpass_warnings = []
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
                surpass_warnings.append(f":- discontiguous {predicate}/{arity}.\n")
    return list(set(surpass_warnings)), rules

def get_prolog_facts(file_dirs):
    facts = []
    for file in file_dirs:
        if os.path.exists(file):
            with open(file, "r") as f:
                facts.extend(f.readlines())
    return facts

def get_pl(rule_file, fact_files, output_file, catch_errors):
    surpass_warnings, rules = get_prolog_rules(rule_file)
    facts = get_prolog_facts(fact_files)
    with open(output_file, "w") as f:
        for warning in surpass_warnings:
            f.write(warning)
        if catch_errors:
            f.write("call_with_catch(Goal, TimeOut) :- catch((call_with_time_limit(60, Goal), TimeOut=false), time_limit_exceeded, (writeln('Query timed out'), TimeOut=true)). \n")
        for rule in rules:
            f.write(rule)
        for fact in facts:
            f.write(fact)
    return None

def get_labeled_data(query_file, catch_errors, use_modified_rules):
    with open(query_file, "r") as f:
        queries = f.readlines()
    outputs = []
    for query in queries:
        query = query.strip()
        if catch_errors:
            try:
                res = janus.query_once(f"call_with_catch({query[:-1]}, TimeOut)")
                print(res)
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
            res = janus.query_once(query)
            output = f"{query}\t{res['truth']}\n"
            print(output)
            outputs.append(output)
        # query = query.strip()[:-1]
        # print(query)
        # #print((f"catch(call_with_time_limit(5, {query}), time_limit_exceeded, writeln('Query timed out'))."))
        # res = janus.query_once(f"catch(call_with_time_limit(5, {query}), time_limit_exceeded, writeln('Query timed out')).")
        # #print(res)
        # output = f"{query}\t{res['truth']}\n"
        # outputs.append(output)
    if not use_modified_rules:
        output_dir = query_file.split(".")[0] + "_label.txt"
    else:
        output_dir = query_file.split(".")[0] + "_mod_label.txt"
    with open(output_dir, "w") as f:
        for output in outputs:
            f.write(output)
    return None



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--level', default='s1', type=str)
    arg_parser.add_argument('--catch_errors', action='store_true')
    arg_parser.add_argument('--use_modified_rules', action='store_true')
    args = arg_parser.parse_args()

    root_dir = f"countries_{args.level}/"
    if args.use_modified_rules:
        get_pl(root_dir + "rules_mod.txt", [root_dir + "facts.txt", root_dir + "train.txt"], root_dir + "countries_mod.pl",
              args.catch_errors)
        janus.consult(root_dir+"countries_mod.pl")
    else:
        get_pl(root_dir+"rules.txt", [root_dir+"facts.txt", root_dir+"train.txt"], root_dir+"countries.pl", args.catch_errors)
        janus.consult(root_dir+"countries.pl")
    print("processing valid.txt")
    get_labeled_data(root_dir+"valid.txt", args.catch_errors, args.use_modified_rules)
    print("processing test.txt")
    get_labeled_data(root_dir+"test.txt", args.catch_errors, args.use_modified_rules)