#!/bin/bash
set -e

cd ..

for i in $(seq 1 $REPETITIONS); do
    python -m bin.test_generation.acme \
        --initial_users 10 \
        --horizon 30 \
        --simulation_ticks_update 3 \
        --autoscaler "hpa50" \
        --objective "underprovisioning" \
        --shape "free" \
        --cores 1,1,1,1,1,1,1,1,1 \
        --output_file "test_acme_perturbed_${i}.json" \
        --perturbation 0.2
done