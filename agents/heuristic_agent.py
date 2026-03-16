import random

class HeuristicAgent:
    def __init__(self):
        self.rank_power = {"4": 0, "5": 1, "6": 2, "7": 3, "Q": 4, "J": 5, "K": 6, "A": 7, "2": 8, "3": 9}
        self.ranks_order = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"]

    def _get_card_strength(self, card_str, vira_str):
        if card_str == "FACEDOWN" or card_str == "" or card_str == "?":
            return -1

        rank, suit = card_str.split('_')
        vira_rank = vira_str.split('_')[0]
        
        vira_idx = self.ranks_order.index(vira_rank)
        manilha_rank = self.ranks_order[(vira_idx + 1) % 10]

        if rank == manilha_rank:
            suit_power = {"DIAMONDS": 1, "SPADES": 2, "HEARTS": 3, "CLUBS": 4}
            return 10 + suit_power[suit]
        
        return self.rank_power[rank]

    def _has_strong_card(self, hand, vira):
        """Retorna True se tiver um 3 ou qualquer Manilha."""
        for card in hand:
            if self._get_card_strength(card, vira) >= 9:
                return True
        return False

    def act(self, state, info):
        legal_actions = info['legal_actions']
        hand = state.get('hand', [])
        vira = state.get('vira', '')
        
        if info.get('waiting_for_mao_de_onze'):
            if self._has_strong_card(hand, vira) and 4 in legal_actions:
                return 4 
            return 5 if 5 in legal_actions else random.choice(legal_actions) 
            
        if state.get('waiting_for_bet', False):
            if self._has_strong_card(hand, vira) and 4 in legal_actions:
                return 4 
            return 5 if 5 in legal_actions else random.choice(legal_actions) 
            
        if 3 in legal_actions and self._has_strong_card(hand, vira):
            if random.random() > 0.5:
                return 3

        play_actions = [a for a in legal_actions if 0 <= a <= 2]
        if play_actions:
            best_action = play_actions[0]
            max_strength = -1
            
            for a in play_actions:
                card = hand[a]
                strength = self._get_card_strength(card, vira)
                if strength > max_strength:
                    max_strength = strength
                    best_action = a
            return best_action

        return random.choice(legal_actions)