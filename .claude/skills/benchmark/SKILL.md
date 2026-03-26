---
description: Run a benchmark tournament and summarize results
tools: Bash, Read
disable-model-invocation: false
---

You are running a benchmark tournament to evaluate agent performance. Follow these steps:

1. Read `play.py` to confirm the available matchup configurations.

2. Ask the user which matchups to run (or default to all):
   - REINFORCE vs Random
   - REINFORCE vs Heuristic
   - REINFORCE vs MCTS
   - MCTS vs Heuristic
   - MCTS vs Random

3. Remind the user to run:
   ```
   python play.py
   ```
   Note: games with MCTS take ~2s each due to tree search. A 10-game tournament with MCTS takes ~20s.

4. After play.py output is provided, format the results as a comparison table:

   | Matchup | P0 Wins | P1 Wins | Win Rate |
   |---------|---------|---------|----------|
   | ...     | ...     | ...     | ...      |

5. Interpret the results:
   - Is REINFORCE beating Heuristic consistently (>85%)?
   - How does REINFORCE compare to MCTS?
   - Flag any anomalies (e.g., worse than random, near-50% vs heuristic).
