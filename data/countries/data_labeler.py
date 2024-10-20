import janus_swi as janus
import re

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
    return surpass_warnings, rules

def get_prolog_facts(file_dir):
    with open(file_dir, "r") as f:
        facts = f.readlines()
    return facts

def get_pl(rule_file, fact_file, output_file):
    surpass_warnings, rules = get_prolog_rules(rule_file)
    facts = get_prolog_facts(fact_file)
    with open(output_file, "w") as f:
        for warning in surpass_warnings:
            f.write(warning)
        for rule in rules:
            f.write(rule)
        for fact in facts:
            f.write(fact)
    return None

def get_labeled_data(query_file):
    with open(query_file, "r") as f:
        queries = f.readlines()
    outputs = []
    for query in queries:
        query = query.strip()
        res = janus.query_once(query)
        output = f"{query}\t{res['truth']}\n"
        outputs.append(output)
    output_dir = query_file.split(".")[0] + "_label.txt"
    with open(output_dir, "w") as f:
        for output in outputs:
            f.write(output)
    return None



if __name__ == "__main__":
    get_pl("rules.txt", "train.txt", "countries.pl")
    janus.consult("countries.pl")
    get_labeled_data("valid.txt")
    get_labeled_data("test.txt")