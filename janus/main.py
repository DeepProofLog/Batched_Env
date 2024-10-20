import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)


janus.consult("ancestor.pl")
# res = janus.query("proof_first([parent(charlie, _1), ancesor(_1, alice)], _T), term_string(_T, T)")
# for d in res:
#     print(d)

state = deque(["ancestor(charlie, alice)"])

while state:
    print("\n\n"+"*"*50)
    print(f'current state is {", ".join(state)}')
    s = state.popleft()
    actions = []
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
            state_list = "["+ s + ", " + ", ".join(state)+"]"
            res = janus.query(f"proof_first({str(state_list)}, _T), term_string(_T, T)")
            for d in res:
                actions.append(d["T"])
            print(f' possible actions are {"; ".join(actions)}')
            # agent = actions[1]
            agent = random.choice(actions)
            clauses = re.findall(r'\w+\(.*?\)', agent)
            state = deque(clauses)
            reward = 0
            print(f'reward is {reward}')
    else:
        for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
            actions.append(d['B'])
        print(f' possible actions are {"; ".join(actions)}')
        #agent = actions[1]
        agent = random.choice(actions)
        clauses = re.findall(r'\w+\(.*?\)', agent)
        for clause in reversed(clauses):
            state.appendleft(clause)
        reward = 0
        print(f'reward is {reward}')
