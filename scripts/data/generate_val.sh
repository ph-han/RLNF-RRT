#!/usr/bin/env sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

uv run python scripts/data/generate_2d.py \
  --split val \
  --num-maps 300 \
  --num-start-goal 12 \
  --width 224 --height 224 \
  --num-points 256 \
  --clearance 2 \
  --step-size 1 \
  --seed 42
