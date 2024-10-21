import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)


def extract_predicates(rule):
    predicate_pattern = r'(\w+)\s*\('
    predicates = re.findall(predicate_pattern, rule)
    return " ".join(predicates)

def process_pl(file):
    res = {}
    with open(file, "r") as f:
        for line in f:
            if line.strip():
                if ":-" in line:
                    predicates = extract_predicates(line.strip())
                    res[predicates] = line.strip()
    return res



janus.consult("ancestor.pl")
# res = janus.query("proof_first([parent(charlie, _1), ancesor(_1, alice)], _T), term_string(_T, T)")
# for d in res:
#     print(d)

rule_construct_dict = process_pl("ancestor.pl")

state = deque(["ancestor(charlie, alice)"])

while state:
#for i in range(5):
    print("\n\n"+"*"*50)
    print(f'current state is {", ".join(state)}')
    s = state.popleft()
    actions = []
    next_states = []
    res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
    if not res["truth"]:
        state = []
        reward = -1
        print("proof failed")
        print(f'reward is {reward}')
    elif res["B"] == "true":
        if not state:
            reward = 1
            print("proof succeeded")
            print(f'reward is {reward}')
        else:
            for d in janus.query(f"prove({s}, _AppliedRule, _NextSubgoals), term_string(_AppliedRule, A)"):
                actions.append(d['A'].split(":-")[0]+".")
            state_list = "["+ s + ", " + ", ".join(state)+"]"
            res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
            for d in res:
                next_states.append(d["T"])
            print(f'possible actions are {"; ".join(actions)}')
            print(f'possible next states are {"; ".join(next_states)}')
            # agent = actions[1]
            agent = random.randint(0, len(actions))
            print(f'agent chose action {actions[agent]}')
            print(f'next state is {next_states[agent]}')
            clauses = re.findall(r'\w+\(.*?\)', next_states[agent])
            state = deque(clauses)
            reward = 0
            print(f'reward is {reward}')
    else:
        for d in janus.query(f"prove({s}, _AppliedRule, _NextSubgoals), term_string(_AppliedRule, A), term_string(_NextSubgoals, N)"):
            next_states.append(d['N'])
            predicates = extract_predicates(d['A'])
            if predicates in rule_construct_dict:
                actions.append(rule_construct_dict[predicates])
            else:
                print("attention, no rule is matched!")
        print(f'possible actions are {"; ".join(actions)}')
        print(f'possible next states are {"; ".join(next_states)}')
        #agent = actions[1]
        agent = random.randint(0, len(actions))
        print(f'agent chose action {actions[agent]}')
        print(f'next state is {next_states[agent]}')
        clauses = re.findall(r'\w+\(.*?\)', next_states[agent])
        for clause in reversed(clauses):
            state.appendleft(clause)
        reward = 0
        print(f'reward is {reward}')
