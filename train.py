import os
import time
import numpy as np
import matplotlib.pyplot as plt
from truco_env.env import TrucoEnv
from truco_env.wrappers import TrucoVectorObservation
from agents.reinforce_agent import ReinforceAgent
from agents.random_agent import RandomAgent

def plot_results(episodes, win_rates, avg_rewards):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(episodes, win_rates, color='blue', label='P0 Win Rate')
    ax1.axhline(y=50.0, color='red', linestyle='--', label='Random Baseline (50%)')
    ax1.set_title('Agent Win Rate vs Random over Episodes')
    ax1.set_ylabel('Win Rate (%)')
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)

    ax2.plot(episodes, avg_rewards, color='green', label='Avg Reward (Window)')
    ax2.set_title('Average Reward per Hand')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    print("Plot saved successfully as 'training_results.png'")

def train(num_episodes=5000):
    base_env = TrucoEnv()
    env = TrucoVectorObservation(base_env)
    
    agent_p0 = ReinforceAgent(lr=1e-3) 
    agent_p1 = RandomAgent()           
    
    wins_p0 = 0
    wins_p1 = 0
    reward_history = []
    
    tracked_episodes = []
    tracked_win_rates = []
    tracked_avg_rewards = []
    
    print("=" * 60)
    print(f"STARTING REINFORCE TRAINING: {num_episodes} MATCHES")
    print("=" * 60)

    start_time = time.time()

    for episode in range(num_episodes):
        state_vector, info = env.reset()
        terminated = False
        truncated = False
        
        episode_reward_p0 = 0.0
        
        while not (terminated or truncated):
            current_player = info['current_player']
            
            if current_player == 0:
                action = agent_p0.act(state_vector, info)
            else:
                action = agent_p1.act(None, info)
                
            next_state_vector, reward, terminated, truncated, next_info = env.step(action)
            
            reward_for_p0 = next_info['reward_p0'] 

            if reward_for_p0 != 0:
                agent_p0.store_reward(reward_for_p0)
                episode_reward_p0 += reward_for_p0
                agent_p0.update_policy()

            state_vector = next_state_vector
            info = next_info
            
        if base_env.current_state['winner'] == 0:
            wins_p0 += 1
        else:
            wins_p1 += 1
            
        reward_history.append(episode_reward_p0)

        if (episode + 1) % 100 == 0:
            win_rate = (wins_p0 / (episode + 1)) * 100
            avg_reward = np.mean(reward_history[-100:])
            
            tracked_episodes.append(episode + 1)
            tracked_win_rates.append(win_rate)
            tracked_avg_rewards.append(avg_reward)
            
            if (episode + 1) % 500 == 0:
                print(f"Episode {episode + 1}/{num_episodes} | P0 Win Rate: {win_rate:.1f}% | Avg Reward (Last 100): {avg_reward:.2f}")

    end_time = time.time()
    
    print("=" * 60)
    print("TRAINING COMPLETED")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Final AI Win Rate: {(wins_p0/num_episodes)*100:.1f}%")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    model_name = f"models/reinforce_{num_episodes}eps_vs_{agent_p1.__class__.__name__}.pth"
    agent_p0.save(model_name)
    print(f"Model weights saved successfully to: {model_name}")
    
    plot_results(tracked_episodes, tracked_win_rates, tracked_avg_rewards)
    
if __name__ == "__main__":
    train(num_episodes=10000)