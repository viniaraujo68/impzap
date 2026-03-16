import random

class RandomAgent:
    def act(self, state, info):
        legal_actions = info['legal_actions']
        return random.choice(legal_actions)