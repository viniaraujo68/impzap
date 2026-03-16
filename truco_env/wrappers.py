import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrucoVectorObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        ranks = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"]
        suits = ["CLUBS", "SPADES", "HEARTS", "DIAMONDS"]
        
        self.card_to_idx = {}
        idx = 0
        for r in ranks:
            for s in suits:
                self.card_to_idx[f"{r}_{s}"] = idx
                idx += 1
                
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(164,), dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(164, dtype=np.float32)
        
        for card in obs.get('hand', []):
            if card in self.card_to_idx:
                vec[self.card_to_idx[card]] = 1.0
                
        for card in obs.get('table_cards', []):
            if card in self.card_to_idx:
                vec[40 + self.card_to_idx[card]] = 1.0
                
        vira = obs.get('vira', '')
        if vira in self.card_to_idx:
            vec[80 + self.card_to_idx[vira]] = 1.0
            
        for card in obs.get('played_cards', []):
            if card in self.card_to_idx:
                vec[120 + self.card_to_idx[card]] = 1.0
            
        score = obs.get('score', [0, 0])
        vec[160] = min(score[0] / 12.0, 1.0)
        vec[161] = min(score[1] / 12.0, 1.0)
        
        bet = obs.get('current_bet_value', 1)
        vec[162] = min(bet / 12.0, 1.0)

        vec[163] = 1.0 if obs.get('waiting_for_mao_de_onze', False) else 0.0
        
        return vec