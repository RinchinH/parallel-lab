#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$HERE/../cuda/jacobi_cuda"

N="${1:-1000000}"
EPS="${2:-1e-8}"
MAXIT="${3:-10000}"

# 5 значений block size (аналог “разной степени параллелизма” на GPU)
BLOCKS=(64 128 256 512 1024)

out_dir="$HERE/../results"
mkdir -p "$out_dir"
csv="$out_dir/timings.csv"

# пересоздаём CSV
echo "res_mode,N,block,eps,maxit,time_sec,iters,residual,rel_err" > "$csv"

# убедимся что бинарник собран
make -C "$HERE/../cuda" clean
make -C "$HERE/../cuda"

for mode in atomic host; do
  for block in "${BLOCKS[@]}"; do
    line="$("$EXE" -n "$N" -eps "$EPS" -k "$MAXIT" --block "$block" --res "$mode" -q)"
    echo "$mode,$N,$block,$EPS,$MAXIT,$line" | tee -a "$csv"
  done
done

echo "Done -> $csv"
