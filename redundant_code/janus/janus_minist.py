import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)

janus.consult("addition_simple.pl")

state = deque(["addition(1)"])

while state:
    print("\n\n"+"*"*50)
    print(f'current state is {", ".join(state)}')
    s = state.popleft()
    #print(s)
    actions = []
    atoms = re.findall(r'\w+\(.*?\)', s)
    if not atoms:
        try:
            res = janus.query_once(f"{s}.")
        except:
            res = {"truth": True, "B": "true"}
    else:
        try:
            res = janus.query_once(f"clause({s}, _B), term_string(_B, B).")
        except:
            res = {"truth": True, "B": "true"}
    #print(res)
    if not res["truth"]:
        state = []
        reward = -1
        print("proof failed")
        print(f'reward is {reward}')
    elif len(res) == 1:
        reward = 1
        print("proof succeeded")
        print(f'reward is {reward}')
    else:
        if res["B"] == "true":
            if not state:
                reward = 1
                print("proof succeeded")
                print(f'reward is {reward}')
            else:
                state_list = "[" + s + ", " + ", ".join(state) + "]"
                res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
                for d in res:
                    print(d)
                    actions.append(d["T"][1:-1])
                print(f' possible actions are {"; ".join(actions)}')
                agent = actions[1]
                # agent = random.choice(actions)
                #clauses = re.findall(r'\w+\(.*?\)', agent)
                atoms = re.findall(r'\w+\(.*?\)', agent)
                constraint_str = agent
                for atom in atoms:
                    constraint_str = constraint_str.replace(atom, "")
                constraints = []
                for i, c in enumerate(constraint_str.split(",")):
                    if c:
                        constraints.append([c, i])
                clauses = atoms
                for c, i in constraints:
                    clauses.insert(i, c)
                # constraints = [c for c in constraint_str.split(",") if c]
                # clauses = atoms + constraints
                state = deque(clauses)
                reward = 0
                print(f'reward is {reward}')
        else:
            for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
                actions.append(d['B'])
            print(f' possible actions are {"; ".join(actions)}')
            if len(actions) == 1:
                agent = actions[0]
            else:
                agent = actions[1]
            # agent = random.choice(actions)
            atoms = re.findall(r'\w+\(.*?\)', agent)
            constraint_str = agent
            for atom in atoms:
                constraint_str = constraint_str.replace(atom, "")
            constraints = []
            for i, c in enumerate(constraint_str.split(",")):
                if c:
                    constraints.append([c, i])
            clauses = atoms
            for c, i in constraints:
                clauses.insert(i, c)
            # clauses = re.findall(r'([a-zA-Z_]+\([^\)]*\))|([^\s,]+)', agent)
            for clause in reversed(clauses):
                state.appendleft(clause)
            reward = 0
            print(f'reward is {reward}')