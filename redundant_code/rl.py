import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces
import numpy as np
import janus_swi as janus
from collections import deque
import re
import random
random.seed(23)

class LogicProofEnv(gym.Env):
    def __init__(self, initial_state):
        super(LogicProofEnv, self).__init__()
        self.state = deque(re.findall(r'\w+\(.*?\)', initial_state))
        self.actions = self.get_actions_prolog(self.state)
        self.action_space = spaces.Discrete(len(self.actions))

    def extract_var(self, state):
        var = re.search(r'_(\d+)', state)
        if var:
            return var.group()
        else:
            return None

    def get_actions_prolog(self, state):
        s = state.popleft()
        actions = []
        res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
        if not res["truth"]:
            return ["False"]
        elif res["B"] == "true":
            if not self.extract_var(s):
                if not state:
                    return ["True"]
            else:
                var = self.extract_var(s)
                s = s.replace(var, "REPLACE")
                res = janus.query_once(f"clause({s}, _B), term_string(_B, B)")
                if not res["truth"]:
                    return ["False"]
                else:
                    for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
                        unification = d['REPLACE']
                        actions.append(", ".join(state).replace(var, unification))
        else:
            for d in janus.query(f"clause({s}, _B), term_string(_B, B)"):
                actions.append(d['B']+",".join(state))
        return actions

    def get_query(self):
        #ToDO: make it close world auto, extract from .pl
        predicates = {("parent", 2), ("ancestor", 2)}
        constants = ["bob", "alice", "charlie", "mary"]
        predicate_random_choice = random.choice(list(predicates))
        predicate, arity = predicate_random_choice[0], predicate_random_choice[1]
        constants = random.sample(constants, arity)
        return predicate+"("+", ".join(constants)+")"

    def reset(self):
        self.state = deque(re.findall(r'\w+\(.*?\)', self.get_query()))
        self.actions = self.get_actions_prolog(self.state)
        self.action_space = spaces.Discrete(len(self.actions))
        return self.state

    def step(self, action):
        self.state = action
        if action == "True":
            return self.state, 1, True, {}
        elif action == "False":
            return self.state, -1, True, {}
        else:
            return self.state, 0, False, {}


janus.consult("ancestor.pl")

initial_state = "ancestor(charlie, alice)"
env = LogicProofEnv(initial_state)

# Set up the PPO agent with a multilayer perceptron (MLP) policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent for 10,000 timesteps
model.learn(total_timesteps=10000)

# Test the trained model
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.render()  # Optional: render the environment
    if done:
        obs, _ = env.reset()

# Close the environment when done
env.close()