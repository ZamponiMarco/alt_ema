#!/bin/bash
set -e

cd ..

python -m bin.model.gen_random_qn \
  --output-folder resources/random_qns/ \
  --num-stations 50 \
  --num-networks 10 \
  --skewness 50 \
  --max-users 300 \
  --k-parameter 5 \
  --min-mu 25 \
  --max-mu 100
