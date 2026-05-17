"""
Statistical analysis of per-game benchmark data.

Reads bench_per_game.csv and produces two outputs:

  bench_stats_marginal.csv:
    For each (tournament_label, p0_name, p1_name): win count, total games,
    win rate, and Wilson 95% CI (lower, upper).

  bench_stats_mcnemar.csv:
    For each pair of agents X, Y that played against the same opponent in
    the same label: number of discordant pairs (b = X wins & Y loses,
    c = X loses & Y wins), McNemar's chi-squared with continuity correction,
    and the two-sided p-value (exact binomial when b+c < 25, chi-squared
    approximation otherwise).

Usage:
    python bench_stats.py --in bench_per_game.csv \
                          --marginal bench_stats_marginal.csv \
                          --paired   bench_stats_mcnemar.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Tuple


Z_95 = 1.959963984540054  # qnorm(0.975)


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k | X ~ Binom(n, p)). Pure stdlib using math.comb."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return total


def wilson_ci(wins: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Return (lower, upper) of the Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_p_value(b: int, c: int) -> float:
    """
    Two-sided p-value for McNemar's test of paired binary outcomes.

    Uses the exact binomial test for small samples (b + c < 25) and the
    continuity-corrected chi-squared approximation otherwise.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        k = min(b, c)
        # Two-sided: 2 * P(B <= k | B ~ Binom(n, 0.5)), clamped to 1.
        p = 2.0 * binom_cdf(k, n, 0.5)
        return min(p, 1.0)
    # Continuity-corrected chi-squared, 1 df.
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Survival function of chi-squared with 1 df = P(X >= chi2) via erfc.
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return p


def mcnemar_chi2_cc(b: int, c: int) -> float:
    """Chi-squared statistic with continuity correction."""
    n = b + c
    if n == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / n


def load_rows(path: str) -> List[dict]:
    """Read the per-game CSV."""
    rows: List[dict] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "tournament_label": r["tournament_label"],
                "p0_name": r["p0_name"],
                "p1_name": r["p1_name"],
                "master_seed": int(r["master_seed"]),
                "game_idx": int(r["game_idx"]),
                "game_seed": int(r["game_seed"]),
                "winner_seat": int(r["winner_seat"]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Stats analysis on per-game CSV.")
    parser.add_argument("--in", dest="in_path", default="bench_per_game.csv")
    parser.add_argument("--marginal", default="bench_stats_marginal.csv")
    parser.add_argument("--paired", default="bench_stats_mcnemar.csv")
    args = parser.parse_args()

    rows = load_rows(args.in_path)
    print(f"Loaded {len(rows)} rows from {args.in_path}.")

    # ---- Marginal analysis: Wilson CI per tournament ----
    # Key: (label, p0, p1) -> list of winner_seat
    tournaments: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for r in rows:
        tournaments[(r["tournament_label"], r["p0_name"], r["p1_name"])].append(r["winner_seat"])

    with open(args.marginal, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament_label", "p0_name", "p1_name",
            "p0_wins", "p1_wins", "ties", "num_games",
            "p0_win_rate_pct", "wilson_lower_pct", "wilson_upper_pct",
            "half_width_pct",
        ])
        for (label, p0, p1), seats in sorted(tournaments.items()):
            n = len(seats)
            p0w = sum(1 for s in seats if s == 0)
            p1w = sum(1 for s in seats if s == 1)
            ties = n - p0w - p1w
            rate = p0w / n
            lo, hi = wilson_ci(p0w, n)
            writer.writerow([
                label, p0, p1, p0w, p1w, ties, n,
                f"{rate*100:.2f}",
                f"{lo*100:.2f}",
                f"{hi*100:.2f}",
                f"{(hi-lo)/2*100:.2f}",
            ])
    print(f"Wrote {args.marginal}.")

    # ---- Paired analysis (McNemar): for each (label, p1), compare all p0 pairs ----
    # Index: (label, p1) -> dict p0 -> { game_seed: winner_seat }
    index: Dict[Tuple[str, str], Dict[str, Dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        key = (r["tournament_label"], r["p1_name"])
        index[key][r["p0_name"]][r["game_seed"]] = r["winner_seat"]

    with open(args.paired, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "tournament_label", "opponent_p1",
            "agent_X", "agent_Y",
            "X_wins", "Y_wins",
            "both_win", "both_lose",
            "b_X_only", "c_Y_only",
            "n_paired",
            "chi2_cc", "p_value", "significant_005",
        ])
        for (label, p1), agents_to_games in sorted(index.items()):
            agents = sorted(agents_to_games.keys())
            for x, y in combinations(agents, 2):
                gx = agents_to_games[x]
                gy = agents_to_games[y]
                common = sorted(set(gx) & set(gy))
                if not common:
                    continue
                a = b = c = d = 0   # 2x2 contingency on paired games
                for seed in common:
                    x_won = (gx[seed] == 0)  # X is P0 in its tournament
                    y_won = (gy[seed] == 0)
                    if x_won and y_won:
                        a += 1
                    elif x_won and not y_won:
                        b += 1
                    elif not x_won and y_won:
                        c += 1
                    else:
                        d += 1
                n_paired = a + b + c + d
                chi2 = mcnemar_chi2_cc(b, c)
                p = mcnemar_p_value(b, c)
                writer.writerow([
                    label, p1, x, y,
                    a + b, a + c,        # marginal wins per agent
                    a, d,                # both-win, both-lose
                    b, c,                # discordant
                    n_paired,
                    f"{chi2:.3f}",
                    f"{p:.4g}",
                    "yes" if p < 0.05 else "no",
                ])
    print(f"Wrote {args.paired}.")

    # ---- Console summary of headline pairings ----
    print()
    print("=" * 75)
    print("Headline McNemar comparisons (vs Heuristic, 5000 games each):")
    print("=" * 75)
    target = ("final_vs_heur", "heuristic")
    if target in index:
        agents_in = index[target]
        candidates = ["hmm_cfr", "hmm", "cfr"]
        present = [a for a in candidates if a in agents_in]
        for x, y in combinations(present, 2):
            gx, gy = agents_in[x], agents_in[y]
            common = sorted(set(gx) & set(gy))
            b = c = a = d = 0
            for seed in common:
                xw = (gx[seed] == 0)
                yw = (gy[seed] == 0)
                if xw and yw: a += 1
                elif xw: b += 1
                elif yw: c += 1
                else: d += 1
            wx = (a + b)
            wy = (a + c)
            n = a + b + c + d
            p = mcnemar_p_value(b, c)
            x_rate = wx / n * 100
            y_rate = wy / n * 100
            print(f"  {x:>8s} ({x_rate:5.2f}%) vs {y:>8s} ({y_rate:5.2f}%): "
                  f"discordant b={b}, c={c} | p = {p:.4g} "
                  f"{'(SIGNIFICANT)' if p < 0.05 else '(n.s.)'}")


if __name__ == "__main__":
    main()
