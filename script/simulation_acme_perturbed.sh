#!/bin/bash
set -e

cd ..

TRAJECTORY_DIR="resources/workloads"
BASE_OUTPUT_DIR="resources/simulation/"

for i in $(seq 1 30); do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$i"
    TRAJECTORY="$TRAJECTORY_DIR/test_acme_perturbed_$i.json"
    USERS=$(jq -r '.users | join(",")' "$TRAJECTORY")
    echo "Running simulation for interval $interval, replica $i"
    echo "Users: $USERS"
    python -m bin.simulation.acme --change_interval 60 --core_update 3 --user_trajectory $USERS --output_dir $OUTPUT_DIR
done
