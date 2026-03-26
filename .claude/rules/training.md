# Training Design Decisions

## Reward Structure
- Per-hand reward: `+CurrentBet` for winner, `-CurrentBet` for loser (typically ±1 to ±3 per hand)
- Rewards are assigned symmetrically at hand resolution (not game end)
- `reward_p0` is available in `info` at every step, including opponent turns
- Collect at ALL steps and use `p0_step_indices` to align with log_probs — do NOT collect only at p0's turns (that misses rewards from opponent folding, etc.)

## Training Curriculum
1. Phase 1 (heuristic): train against `HeuristicAgent` until 65% rolling win rate (500-episode window)
2. Phase 2 (mixed): 50% episodes against frozen self-play snapshot, 50% against `HeuristicAgent`
   - Rationale: pure self-play causes catastrophic forgetting of heuristic-beating strategies
   - Frozen snapshot updated every 500 episodes

## Evaluation
- Eval interval and window scale automatically with `num_episodes`: `EVAL_INTERVAL = max(500, num_episodes // 20)`, `WINDOW = EVAL_INTERVAL * 2`
- Each eval checkpoint: 300 games vs `RandomAgent` + 300 games vs `HeuristicAgent`
- MCTS agent is used only as a benchmark in `play.py`, not during training (too slow: ~0.4s/game vs ~0.005s)

## Hyperparameters
- `lr = 1e-3`, `gamma = 0.99`, `ema_alpha = 0.05`
- `hidden_size = 128`, `input_size = 164`, `output_size = 9`
- Self-play mix: `SELF_PLAY_MIX = 0.5`
- Default `num_episodes = 100_000` (call with any value; intervals scale automatically)

## REINFORCE Performance Ceiling
- 10k episodes: ~26% vs MCTS, ~84% vs Heuristic
- 50k episodes: ~26% vs MCTS (no improvement) — confirmed plateau
- Ceiling is architectural: static policy with no lookahead cannot close the gap against a search algorithm
- Next step: CFR (see algorithms.md)

## Known Issues Fixed
- Within-episode std normalization was dropped — normalizing a single episode's monotonically-discounted return sequence forces early steps of winning games into negative advantage
- Reward collection was previously limited to p0's action steps — fixed to collect all steps
- Raise ladder bug in engine: `requestBet()` was resetting `PendingBet = 3` instead of next value above `CurrentBet`, and incorrectly updating `CurrentBet` during re-raises (only `acceptBet()` should update it)
