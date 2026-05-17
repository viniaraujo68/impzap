#!/bin/bash
# Final analysis pipeline: merge -> stats -> summary.
set -e
cd "$(dirname "$0")"

if [ ! -f bench_per_game_hmmcfr_v8.csv ]; then
  echo "ERROR: bench_per_game_hmmcfr_v8.csv not found. Run bench_csv_hmmcfr_v8.py first."
  exit 1
fi

# 1. Merge v9-bench (non-HMM+CFR rows) + v8-rerun (HMM+CFR rows)
./merge_csv.sh

# 2. Run Wilson CI + McNemar
./.venv/bin/python bench_stats.py \
  --in bench_per_game_final.csv \
  --marginal bench_stats_marginal_final.csv \
  --paired bench_stats_mcnemar_final.csv

# 3. Print headline numbers for table updates
echo
echo "===================================================================="
echo "Tabela 7.1 ('final_vs_heur' and 'arch_*'): values for thesis update"
echo "===================================================================="
python3 - << 'EOF'
import csv
def wilson(p, n):
    import math
    z = 1.96
    den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return c-h, c+h, h

with open("bench_stats_marginal_final.csv") as f:
    rows = list(csv.DictReader(f))

want = [
    ("arch_vs_fold",   "always_fold"),
    ("final_vs_heur",  "heuristic"),
    ("final_vs_rand",  "random"),
    ("arch_vs_raise",  "always_raise"),
    ("deterministic",  "deterministic"),
]
print(f"{'Tournament':<22s} {'Agent':<12s} {'Rate':>7s}  {'Wilson CI':<18s}  {'±half':>7s}")
print("-" * 75)
for (label, p1) in want:
    for r in rows:
        if r["tournament_label"] != label or r["p1_name"] != p1:
            continue
        rate = float(r["p0_win_rate_pct"])
        lo = float(r["wilson_lower_pct"])
        hi = float(r["wilson_upper_pct"])
        hw = float(r["half_width_pct"])
        print(f"{label:<22s} {r['p0_name']:<12s} {rate:>6.2f}% [{lo:5.2f}, {hi:5.2f}] ±{hw:5.2f}")
    print()
EOF
