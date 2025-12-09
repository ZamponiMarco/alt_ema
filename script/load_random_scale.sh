#!/bin/bash
set -e

cd ..

python -m bin.test_generation.random \
  --qn-folder resources/random_qns/ \
  --output-folder resources/workloads/ \
  --horizon 30 \
  --initial_users 10 \
  --objective underprovisioning \
  --shape free \
  --time_limit 600
