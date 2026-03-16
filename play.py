import time
from truco_env.env import TrucoEnv
from agents.heuristic_agent import HeuristicAgent
from agents.random_agent import RandomAgent

def translate_action(action, state):
    is_mao_de_onze = state.get('waiting_for_mao_de_onze', False)
    
    if action == 3:
        return "RAISED"
    if action == 4:
        return "AGREED TO PLAY (MÃO DE 11)" if is_mao_de_onze else "ACCEPTED RAISE"
    if action == 5:
        return "REFUSED TO PLAY (MÃO DE 11)" if is_mao_de_onze else "FOLDED"
    
    if 0 <= action <= 2:
        hand = state.get('hand', [])
        card_name = hand[action] if action < len(hand) else "?"
        return f"PLAYED {card_name} (Open)"
        
    if 6 <= action <= 8:
        real_index = action - 6
        hand = state.get('hand', [])
        card_name = hand[real_index] if real_index < len(hand) else "?"
        return f"PLAYED {card_name} FACEDOWN"
        
    return f"UNKNOWN ACTION {action}"

def get_agent_name(agent):
    return getattr(agent, 'name', agent.__class__.__name__)

def play_verbose_match(env, agent_p0, agent_p1):
    state, info = env.reset()
    terminated = False
    truncated = False
    turn = 1
    hand_number = 1

    name_p0 = get_agent_name(agent_p0)
    name_p1 = get_agent_name(agent_p1)

    print("="*60)
    print(f"📺 WATCHING A SINGLE MATCH: {name_p0} (P0) vs {name_p1} (P1)")
    print("="*60)

    while not (terminated or truncated):
        current_player = state['current_player']
        
        print(f"\n[Hand {hand_number} | Turn {turn}] Score: P0 [{state['score'][0]}] x [{state['score'][1]}] P1")
        print(f"Vira: {state['vira']} | Current Bet: {state['current_bet_value']}")
        print(f"Table: {state['table_cards']}")
        print(f"Player {current_player}'s Hand: {state['hand']}")
        
        if info.get('waiting_for_mao_de_onze'):
            print(f"MÃO DE 11! Player {current_player} is deciding whether to play or flee.")
        
        if current_player == 0:
            action = agent_p0.act(state, info)
            player_name = f"P0 ({name_p0})"
        else:
            action = agent_p1.act(state, info)
            player_name = f"P1 ({name_p1})"
            
        print(f"{player_name}: {translate_action(action, state)}")

        next_state, reward, terminated, truncated, next_info = env.step(action)

        if reward != 0:
            print("-" * 60)
            print(f"⭐ HAND ENDED! Reward delivered: {reward:+.1f}")
            print(f"⭐ NEW SCORE: P0 [{next_state['score'][0]}] x [{next_state['score'][1]}] P1")
            print("-" * 60)
            hand_number += 1

        state = next_state
        info = next_info
        turn += 1

    print("="*60)
    print(f"🏆 MATCH ENDED! WINNER: PLAYER {state['winner']}")
    print("="*60)


def simulate_tournament(env, agent_p0, agent_p1, num_games=1000):
    wins_p0 = 0
    wins_p1 = 0
    
    name_p0 = get_agent_name(agent_p0)
    name_p1 = get_agent_name(agent_p1)
    
    print(f"\nSTARTING TOURNAMENT: {num_games} MATCHES")
    print(f"P0: {name_p0} | P1: {name_p1}")
    start_time = time.time()

    for game in range(num_games):
        state, info = env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            current_player = state['current_player']
            
            if current_player == 0:
                action = agent_p0.act(state, info)
            else:
                action = agent_p1.act(state, info)
                
            state, reward, terminated, truncated, info = env.step(action)
            
        if state['winner'] == 0:
            wins_p0 += 1
        elif state['winner'] == 1:
            wins_p1 += 1

    end_time = time.time()
    
    win_rate_p0 = (wins_p0 / num_games) * 100
    win_rate_p1 = (wins_p1 / num_games) * 100
    
    print("="*50)
    print("🏆 TOURNAMENT FINAL RESULTS 🏆")
    print("="*50)
    print(f"Simulation Time: {end_time - start_time:.2f} seconds")
    print(f"Wins P0 ({name_p0}): {wins_p0} ({win_rate_p0:.1f}%)")
    print(f"Wins P1 ({name_p1}): {wins_p1} ({win_rate_p1:.1f}%)")

def main():
    env = TrucoEnv()
    
    agent_p0 = HeuristicAgent()
    agent_p1 = RandomAgent() 
    
    play_verbose_match(env, agent_p0, agent_p1)
    
    simulate_tournament(env, agent_p0, agent_p1, num_games=1000)

if __name__ == "__main__":
    main()