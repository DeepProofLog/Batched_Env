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

            # # raise an error if the vars in the body or head are not letters (only the arguments)
            # vars_head = re.findall(r'\b([a-zA-Z_]\w*)\b', head)
            # vars_body = re.findall(r'\b([a-zA-Z_]\w*)\b', body)

            # if len(vars_head) > 1 and not all(var.isalpha() for var in vars_head[1:]):
            #      # Find the specific non-alpha argument for a clearer error
            #      invalid_arg = next((var for var in vars_head[1:] if not var.isalpha()), None)
            #      raise ValueError(f"Error in rule: {rule}. Argument '{invalid_arg}' in head '{head}' must be letters only.")

            # if len(vars_body) > 1 and not all(var.isalpha() for var in vars_body[1:]):
            #      invalid_arg = next((var for var in vars_body[1:] if not var.isalpha()), None)
            #      raise ValueError(f"Error in rule: {rule}. Argument '{invalid_arg}' in body '{body}' must be letters only.")

            # if not all(var.isalpha() for var in vars_head):
            #     raise ValueError(f"Error in rule: {rule}. Variables in head must be letters.")
            # if not all(var.isalpha() for var in vars_body):
            #     raise ValueError(f"Error in rule: {rule}. Variables in body must be letters.")
            # if the vars are not in capital, make them capital

            body = re.sub(r'\b([a-z])\b', lambda x: x.group(1).upper(), body)
            head = re.sub(r'\b([a-z])\b', lambda x: x.group(1).upper(), head)
            # remove '_' from the predicates starting with '_' in the body and head
            head = re.sub(r'\b_([a-zA-Z]\w*)\b', lambda x: x.group(1), head)
            body = re.sub(r'\b_([a-zA-Z]\w*)\b', lambda x: x.group(1), body)
            print(f"Rule: {head} :- {body}.")
            # for each predicate in the body, add it to the predicates list
            prolog_rule = f"{head} :- {body}.\n"
            rules.append(prolog_rule)
            matches = re.findall(r'(\b\w+)\(([^)]*)\)', prolog_rule)
            predicates_with_arity = [(predicate, len(args.split(','))) for predicate, args in matches]
            for predicate, arity in predicates_with_arity:
                # remove '_' from the predicates starting with '_'
                if predicate.startswith("_"):
                    predicate = predicate[1:]
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
    facts_new = []
    for f in facts:
        matches = re.findall(r'(\b\w+)\(([^)]*)\)', f)
        # remove '_' from the predicates starting with '_' in the body and head
        f = re.sub(r'\b_([a-zA-Z]\w*)\b', lambda x: x.group(1), f)
        # Given the format pred(arg1,arg2) in facts, add "" to the args: pred("arg1","arg2")
        f = re.sub(r'(\w+)\(([^)]*)\)', lambda x: f'{x.group(1)}("{x.group(2).replace(",", '","')}")', f)
        facts_new.append(f)
        predicates_with_arity = [(predicate, len(args.split(','))) for predicate, args in matches]
        for predicate, arity in predicates_with_arity:
            # remove '_' from the predicates starting with '_'
            if predicate.startswith("_"):
                predicate = predicate[1:]
            if predicate not in predicates:
                predicates[predicate] = arity
    # add a \n to the last fact
    facts_new[-1] = facts_new[-1] + "\n"
    return predicates, facts_new

def get_pl(rule_file, fact_files, output_file, use_tabling=False):
    """
    Generates a Prolog file from the given rule and fact files.
    The generated Prolog file contains dynamic and discontiguous predicates,
    as well as the rules and facts from the input files.
    Args:
        rule_file (str): Path to the file containing Prolog rules.
        fact_files (list): List of paths to files containing Prolog facts.
        output_file (str): Path to the output Prolog file.
        use_tabling (bool): If True, adds table directives for predicates.
    """
    predicates_1, rules = get_prolog_rules(rule_file)
    predicates_2, facts = get_prolog_facts(fact_files)
    predicates = {**predicates_1, **predicates_2}
    outputs = []
    full_facts = []
    with open(output_file, "w") as f:
        for predicate, arity in predicates.items():
            f.write(f":- dynamic {predicate}/{arity}.\n")
        for predicate, arity in predicates.items():
            f.write(f":- discontiguous {predicate}/{arity}.\n")
        if use_tabling:
            for predicate, arity in predicates.items():
                f.write(f":- table {predicate}/{arity}.\n")
        for fact in facts:
            f.write(fact)
        for rule in rules:
            f.write(rule)

        # add this at the end of the file
        add_str = '''one_step_list([Goal|RestGoals], NewGoalsList) :-
            findall(NewGoals,
                (
                    clause(Goal, Body),
                    body_to_list(Body, BodyList),
                    append(BodyList, RestGoals, NewGoals)
                ),
                AllNewGoals),
            (
                AllNewGoals = [] ->
                    NewGoalsList = [false]
                ;
                    NewGoalsList = AllNewGoals
            ).
        body_to_list(true, []) :- !.
        body_to_list((A,B), [A|Rest]) :- !, body_to_list(B, Rest).
        body_to_list(A, [A]).'''
        f.write(add_str)

    return None


if __name__ == "__main__":
    
    '''
    Obtains a Prolog file from the rules and facts files.    
    '''

    dataset = 'family'
    current_dir = os.getcwd()
    root_dir = f"{current_dir}/data/{dataset}/"

    get_pl(root_dir+"rules.txt", [root_dir+"facts.txt", root_dir+"train.txt"], root_dir+ dataset+".pl",)


