#!/usr/bin/env sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

uv run python scripts/data/generate_2d.py \
  --split test \
  --split-name test_circle \
  --num-maps 300 \
  --num-start-goal 12 \
  --width 224 --height 224 \
  --num-points 256 \
  --clearance 2 \
  --step-size 1 \
  --min-start-goal-dist 40 \
  --max-start-goal-dist 170 \
  --map-style circle_hard \
  --min-circles 28 \
  --max-circles 64 \
  --min-circle-radius 5 \
  --max-circle-radius 20 \
  --seed 144
