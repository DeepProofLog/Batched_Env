import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)


janus.consult("ancestor.pl")

state = deque(["ancestor(charlie, alice)"])

def extract_var(text):
    var = re.search(r'_(\d+)', text)
    if var:
        return var.group()
    else:
        return None

while state:
    print(state)
    s = state.popleft()
    actions = []
    res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
    if not res["truth"]:
        state = []
        reward = -1
        print("proof failed")
    elif res["B"] == "true":
        if not extract_var(s):
            if not state:
                reward = 1
                print("proof succeeded")
        else:
            var = extract_var(s)
            s = s.replace(var, "REPLACE")
            res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
            if not res["truth"]:
                state = []
                reward = -1
                print("proof failed")
            else:
                for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
                    unification = d['REPLACE']
                    actions.append(", ".join(state).replace(var, unification))
                print(actions)
                #agent = actions[0]
                agent = random.choice(actions)
                clauses = re.findall(r'\w+\(.*?\)', agent)
                state = deque(clauses)
                reward = 0
    else:
        for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
            actions.append(d['B'])
        print(actions)
        #agent = actions[1]
        agent = random.choice(actions)
        clauses = re.findall(r'\w+\(.*?\)', agent)
        for clause in reversed(clauses):
            state.appendleft(clause)
        reward = 0
