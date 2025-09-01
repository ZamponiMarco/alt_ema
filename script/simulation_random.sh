#!/bin/bash
set -e

cd ..

BASE_OUTPUT_DIR="resources/simulation/random"
INTERVAL=20

for i in {0..49}; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$i"
    echo "Running simulation for id $i, interval $INTERVAL, output directory $OUTPUT_DIR"
    python -m bin.simulation.random --change_interval $INTERVAL --core_update 3 --id $i --output_dir "$OUTPUT_DIR"
done