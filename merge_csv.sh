#!/bin/bash
# Merge bench_per_game.csv (v9-backed) with bench_per_game_hmmcfr_v8.csv
# Output: bench_per_game_final.csv with HMM+CFR rows replaced by v8-backed version.
set -e

cd "$(dirname "$0")"

OUT=bench_per_game_final.csv

head -1 bench_per_game.csv > $OUT
# All non-HMM+CFR rows from v9 bench
awk -F, 'NR>1 && $2!="hmm_cfr" && $3!="hmm_cfr"' bench_per_game.csv >> $OUT
# All HMM+CFR rows from v8 rerun
awk -F, 'NR>1' bench_per_game_hmmcfr_v8.csv >> $OUT

wc -l $OUT
echo "Merged: rows above"
