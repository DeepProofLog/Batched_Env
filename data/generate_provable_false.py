import os
import janus_swi as janus
import re
import argparse


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
    with open(output_file, "w") as f:
        for predicate, arity in predicates.items():
            f.write(f":- discontiguous {predicate}/{arity}.\n")
        if use_tabling:
            f.write(":- table locatedInCR/2.\n")
        for fact in facts:
            f.write(fact)
        if catch_errors:
            f.write("call_with_catch(Goal, TimeOut) :- catch((call_with_time_limit(60, Goal), TimeOut=false), time_limit_exceeded, (writeln('Query timed out'), TimeOut=true)). \n")
        for rule in rules:
            f.write(rule)
    return None

def get_labeled_data(query_file, catch_errors, use_modified_rules):
    with open(query_file, "r") as f:
        queries = f.readlines()
    outputs = []
    for query in queries:
        query = query.strip()
        print(query)
        # res = janus.query(query)
        # for d in res:
        #     print(d)
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
            res = janus.query_once(query)
            #for d in res:
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

    regions = ['oceania', 'asia', 'europe', 'africa', 'americas']

    root_dir = "countries_s2/"
    file_dir = root_dir + "train.txt"
    knowledge_dir = root_dir+"countries.pl"
    janus.consult(knowledge_dir)

    outputs = []
    provable_true = 0

    with open(file_dir, "r") as f:
        queries = f.readlines()
        for q in queries:
            q = q.strip()
            matches = re.findall(r'(\b\w+)\(([^)]*)\)', q)
            for m in matches:
                predicate = m[0]
                if predicate == "locatedInCR":
                    #print(q)
                    provable_true += 1
                    country, region = m[1].split(",")
                    collapses = [f'locatedInCR({country},{r}).' for r in regions if r != region]
                    #print(collapses)
                    for c in collapses:
                        res = janus.query_once(c)
                        if res['truth']:
                            output = f"{c}\n"
                            #print(output)
                            outputs.append(output)

    of_dir = file_dir.split(".")[0] + "_provable_false.txt"
    with open(of_dir, "w") as f:
        for output in outputs:
            f.write(output)

    print(f'Provable True: {provable_true}')