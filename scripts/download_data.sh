#!/usr/bin/env bash
# Downloads the three WANDS CSVs (MIT licence, github.com/wayfair/WANDS) into data/raw/.
set -euo pipefail
mkdir -p data/raw
base=https://raw.githubusercontent.com/wayfair/WANDS/main/dataset
for f in query label product; do
  curl -sSL -o "data/raw/$f.csv" "$base/$f.csv"
done
wc -l data/raw/*.csv
