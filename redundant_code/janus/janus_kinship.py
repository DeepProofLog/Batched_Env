import janus_swi as janus
import re
import random
random.seed(20)

janus.consult("kinship.pl")

state = "[aunt(5,76)]"
# state = "[aunt(5,76),father(12,34)]"

# janus.consult("wn18rr.pl")
# state = "[member_of_domain_usage(06845599,03754979)]"

counter = 0
while counter<20:
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
        agent = random.choice(actions)
        state = agent
        counter += 1