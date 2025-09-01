#!/bin/bash
set -e

cd ..

autoscalers=("hpa50" "hpa60" "step1" "step2")
objectives=("underprovisioning" "overprovisioning")
shapes=("free" "spike" "sawtooth" "ramp")

for autoscaler in "${autoscalers[@]}"; do
    for objective in "${objectives[@]}"; do
        for shape in "${shapes[@]}"; do
            echo "Running with autoscaler=${autoscaler}, objective=${objective}, shape=${shape}"
            python -m bin.test_generation.two_tier \
                --initial_users 10 \
                --horizon 24 \
                --simulation_ticks_update 3 \
                --autoscaler "${autoscaler}" \
                --objective "${objective}" \
                --shape "${shape}" \
                --cores 1,1,1 \
                --output_file "${autoscaler}_${objective}_${shape}.json"
        done
    done
done