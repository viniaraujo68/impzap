---
description: Retrain the REINFORCE agent from scratch
tools: Bash, Read
disable-model-invocation: false
---

You are helping retrain the REINFORCE agent. Follow these steps:

1. Read `train.py` to confirm the current hyperparameters (lr, gamma, ema_alpha, num_episodes, SELF_PLAY_MIX, WINDOW).

2. Present the current hyperparameters to the user and ask if they want to change anything before starting.

3. Once confirmed, remind the user to run:
   ```
   python train.py
   ```
   (Do not run it yourself — training takes 10+ minutes.)

4. After training completes, read `training_results.png` and the terminal output to summarize:
   - Final eval vs Random and vs Heuristic win rates
   - Whether catastrophic forgetting occurred (look for consistent downward trend in eval vs Heuristic)
   - Episode at which self-play activated
   - Whether avg return stayed positive (should be ~+4 to +6 for a winning agent)
