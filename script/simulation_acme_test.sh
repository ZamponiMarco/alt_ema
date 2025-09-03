#!/bin/bash
set -e

cd ..

OUTPUT_DIR="resources/simulation/acme"

for interval in {60..60..1}; do
    for i in {1..10}; do
        echo "Running simulation for interval $interval, replica $i"
        python -m bin.simulation.acme_test --change_interval $interval --core_update 3 --index $i --output_dir $OUTPUT_DIR
    done
done