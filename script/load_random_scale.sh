#!/bin/bash
set -e

cd ..

autoscalers=("hpa50" "step1" "step2" "step3")

# Configuration 1: 50 stations (first feasible only)
for autoscaler in "${autoscalers[@]}"; do
    echo "Generating loads for config1 with autoscaler: $autoscaler (first feasible only)"
    python -m bin.test_generation.random \
      --qn-folder resources/random_qns_config1/ \
      --output-folder resources/workloads_config1_${autoscaler}/ \
      --horizon 30 \
      --initial_users 10 \
      --objective underprovisioning \
      --shape free \
      --autoscaler ${autoscaler} \
      --time_limit 600 \
      --first-feasible
done

# Configuration 2: 30 stations (first feasible only)
for autoscaler in "${autoscalers[@]}"; do
    echo "Generating loads for config2 with autoscaler: $autoscaler (first feasible only)"
    python -m bin.test_generation.random \
      --qn-folder resources/random_qns_config2/ \
      --output-folder resources/workloads_config2_${autoscaler}/ \
      --horizon 30 \
      --initial_users 10 \
      --objective underprovisioning \
      --shape free \
      --autoscaler ${autoscaler} \
      --time_limit 600 \
      --first-feasible
done

# Configuration 3: 15 stations (first feasible only)
for autoscaler in "${autoscalers[@]}"; do
    echo "Generating loads for config3 with autoscaler: $autoscaler (first feasible only)"
    python -m bin.test_generation.random \
      --qn-folder resources/random_qns_config3/ \
      --output-folder resources/workloads_config3_${autoscaler}/ \
      --horizon 30 \
      --initial_users 10 \
      --objective underprovisioning \
      --shape free \
      --autoscaler ${autoscaler} \
      --time_limit 600 \
      --first-feasible
done

# Configuration 4: 10 stations (first feasible only)
for autoscaler in "${autoscalers[@]}"; do
    echo "Generating loads for config4 with autoscaler: $autoscaler (first feasible only)"
    python -m bin.test_generation.random \
      --qn-folder resources/random_qns_config4/ \
      --output-folder resources/workloads_config4_${autoscaler}/ \
      --horizon 30 \
      --initial_users 10 \
      --objective underprovisioning \
      --shape free \
      --autoscaler ${autoscaler} \
      --time_limit 600 \
      --first-feasible
done
