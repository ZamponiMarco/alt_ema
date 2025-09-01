#!/bin/bash
set -e

cd ..

python -m bin.test_generation.acme \
    --initial_users 10 \
    --horizon 30 \
    --simulation_ticks_update 3 \
    --autoscaler "hpa50" \
    --objective "underprovisioning" \
    --shape "free" \
    --cores 1,1,1,1,1,1,1,1,1 \
    --output_file "test.json"