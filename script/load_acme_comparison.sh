#!/bin/bash
set -e

cd ..

INITIAL_USERS=10
HORIZON=60
SIMULATION_TICKS_UPDATE=3
CORES=1,1,1,1,1,1,1,1,1
TIME_LIMIT=300
REPETITIONS=10

for i in $(seq 1 $REPETITIONS); do
    python -m bin.test_generation.acme \
        --initial_users $INITIAL_USERS \
        --horizon $HORIZON \
        --simulation_ticks_update $SIMULATION_TICKS_UPDATE \
        --autoscaler "hpa50" \
        --objective "underprovisioning" \
        --shape "free" \
        --cores $CORES \
        --output_file "test_opt_${i}.json" \
        --time_limit $TIME_LIMIT \
        --tolerance 0 \
        --alpha 1 \
        --beta 0
done

for i in $(seq 1 $REPETITIONS); do
    python -m bin.test_generation.acme_ga \
        --initial_users $INITIAL_USERS \
        --horizon $HORIZON \
        --output_file "test_ga_${i}.json" \
        --cores $CORES \
        --time_limit $TIME_LIMIT
done

for i in $(seq 1 $REPETITIONS); do
    python -m bin.test_generation.acme_random \
        --horizon $HORIZON \
        --initial_users $INITIAL_USERS \
        --output_file "test_random_${i}.json" \
        --cores $CORES \
        --time_limit $TIME_LIMIT
done