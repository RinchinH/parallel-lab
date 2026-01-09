#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXE="$HERE/../openmp/jacobi_openmp"

N="${1:-4000}"
EPS="${2:-1e-8}"
MAXIT="${3:-10000}"
MODE="${4:-fast}"     # fast | dense

out_dir="$HERE/../results"
mkdir -p "$out_dir"
csv="$out_dir/timings.csv"

echo "threads,N,mode,eps,maxit,time_sec,iters,residual,rel_err" > "$csv"

for t in 1 2 4 8 16; do
  export OMP_NUM_THREADS="$t"
  line="$("$EXE" -n "$N" -eps "$EPS" -k "$MAXIT" --$MODE -q)"
  echo "$t,$N,$MODE,$EPS,$MAXIT,$line" | tee -a "$csv"
done

echo "Done -> $csv"
