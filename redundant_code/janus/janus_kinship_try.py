import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)

def extract_var(state: str)-> list:
    '''Extract unique variables from a state: start with uppercase letter or underscore'''
    pattern = r'\b[A-Z_][a-zA-Z0-9_]*\b'
    vars = re.findall(pattern, state)
    return list(dict.fromkeys(var for var in vars if var != "_"))


def variable_mapping():

    with open("kinship.pl", 'r') as f:
        lines = f.readlines()

    rules = {}

    for line in lines:
        line = line.strip()
        rule_pattern = re.compile(r"^\s*(?!\s*:-)(.*)\s*:-\s*(.*)\s*$")
        match = rule_pattern.match(line)
        if match:
            head = match.group(1).strip()
            body = match.group(2).strip()
            head_predicate = head.split("(")[0].strip()
            if head_predicate == "safe_call" or head_predicate == "proof_first":
                continue
            head_args = [arg.strip() for arg in head.split("(")[1].rstrip(")").split(",")]
            body_atoms = re.findall(r"\w+\([a-zA-Z_0-9, ]+\)", body)
            body_predicates = [atom.split("(")[0].strip() for atom in body_atoms]
            body_args = [[arg.strip() for arg in atom.strip().split("(")[1].rstrip(")").split(",")] for atom in body_atoms ]
            rule_signature = f"{head_predicate},{','.join(body_predicates)}"
            if rule_signature in rules:
                rule_signature = rule_signature+"1"
            rules[rule_signature] = {}
            for k in range(len(head_args)):
                h_arg = head_args[k]
                for i in range(len(body_args)):
                    for j in range(len(body_args[i])):
                        if h_arg == body_args[i][j]:
                            if k not in rules[rule_signature]:
                                rules[rule_signature][k] = []
                            rules[rule_signature][k].append([i, j])
    return rules

# TODO: need to find a way to keep the vaiables in the head (a better way than this function)
# TODO: anyway to control new-involved var not equal to existing?
def reset_var(head, body, var_map):
    '''Reset the variable in the body of the rule'''
    head_predicate = head.split("(")[0].strip()
    head_args = [arg.strip() for arg in head.split("(")[1].rstrip(")").split(",")]
    body_atoms = re.findall(r"\w+\([a-zA-Z_0-9, ]+\)", body)
    body_predicates = [atom.split("(")[0].strip() for atom in body_atoms]
    body_args = [[arg.strip() for arg in atom.strip().split("(")[1].rstrip(")").split(",")] for atom in body_atoms]
    rule_signature = f"{head_predicate},{','.join(body_predicates)}"
    rule_var_map = var_map[rule_signature]
    if rule_signature+"1" in var_map:
        atom0_args, atom1_args = body_args
        common = set(atom0_args) & set(atom1_args)
        common_element = common.pop()
        atom0_id = atom0_args.index(common_element)
        atom1_id = atom1_args.index(common_element)
        ids = [value[0] for value in var_map[rule_signature].values()]
        atom0_match, atom1_match = False, False
        for atom_id, position_id in ids:
            if atom_id == 0 and position_id != atom0_id:
                atom0_match = True
            if atom_id == 1 and position_id != atom1_id:
                atom1_match = True
        if not (atom0_match and atom1_match):
            rule_var_map = var_map[rule_signature+"1"]
    for i in range(len(head_args)):
        h_arg = head_args[i]
        if h_arg[0] == "_":
            for atom_id, position_id in rule_var_map[i]:
                body_args[atom_id][position_id] = h_arg
    body_atoms_new = [f"{predicate}({",".join(args)})" for predicate, args in zip(body_predicates, body_args)]
    return ",".join(body_atoms_new)





janus.consult("kinship.pl")

#state = deque(["aunt(5,76)"])
state = "[aunt(5,76)]"

# res = janus.query_once("one_step_list([father(_1,76), sister(5,_1)], _NewGoalsList), term_string(_NewGoalsList, NewGoalsList)")
# print(res)
# res = janus.query_once(f"one_step_list([{state}], _NewGoalsList), term_string(_NewGoalsList, NewGoalsList)")
# print(res)

# var_mapping = variable_mapping()
counter = 0
while True:
    print("\n\n"+"*"*50)
    print(f'current state is {state}')
    res = janus.query_once(f"one_step_list({state}, _NewGoalList), term_string(_NewGoalList, NewGoalList)")
    print(res)
    actions = re.findall(r'\[[^\[\]]*\]', res['NewGoalList'])
    if any(a=="[]" for a in actions):
        print("proof succeeded")
        break
    if any(a=="[false]" for a in actions):
        print("proof failed")
        break
    else:
        if counter == 0:
            agent = actions[1]
        if counter == 1:
            agent = actions[0]
        state = agent
        counter += 1



    # res_dict = {}
    # for d in res:
    #     if "truth" in res_dict:
    #         res_dict["truth"].append(d['truth'])
    #     else:
    #         res_dict["truth"] = [d['truth']]
    #     if "B" in d:
    #         body = d['B']
    #     else:
    #         body = None
    #     if "B" in res_dict:
    #         res_dict["B"].append(body)
    #     else:
    #         res_dict["B"] = [body]
    # print('Res_dict:', res_dict)
    #
    #
    # # If no matching fact
    # if res_dict == {} or res_dict["truth"] == [False]:
    #     print(f'There is nothing matching')
    #     print(f'Proof failed')
    #     break
    # else:
    #     if any(t and (b=='true') for t, b in zip(res_dict["truth"], res_dict["B"])):
    #         if (not state) and (not extract_var(s)):
    #             print(f'It is a fact')
    #             print(f'Proof succeeded')
    #             break
    #         else:
    #             state_list = "[" + s + ", " + ", ".join(state) + "]"
    #             res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
    #             for d in res:
    #                 actions.append(d["T"][1:-1])
    #             for b in res_dict["B"]:
    #                 if b != "true":
    #                     if state:
    #                         b_reset = reset_var(s, b, var_mapping)
    #                         actions.append(b_reset + "," + ",".join(state))
    #                     else:
    #                         actions.append(b)
    #             agent = actions[0]
    #             clauses = re.findall(r'\w+\(.*?\)', agent)
    #             state = deque(clauses)
    #     else:
    #         if state:
    #             actions = [reset_var(s, b, var_mapping) + ", " + ", ".join(state) for b in res_dict["B"]]
    #         else:
    #             actions = res_dict["B"]
    #         print(f' possible actions are {"; ".join(actions)}')
    #         agent = actions[0]
    #         clauses = re.findall(r'\w+\(.*?\)', agent)
    #         state = deque(clauses)