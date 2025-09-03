#!/bin/bash
set -e

cd ..

TRAJECTORY="10.0,10.0,6.97528358,11.39897612,2.13968947,8.12343914,3.08462409,10.0230566,10.3058977,19.75770231,15.94421605,11.88522312,21.74601217,18.19541953,24.4616582,28.3420229,32.67213379,29.70802123"
OUTPUT_DIR="resources/simulation/acme"

for interval in {20..20..1}; do
    for i in {1..1}; do
        echo "Running simulation for interval $interval, replica $i"
        python -m bin.simulation.acme --change_interval $interval --core_update 3 --user_trajectory $TRAJECTORY --output_dir $OUTPUT_DIR
    done
done

TRAJECTORY="10.0,10.0,3.8627608346709454,1.0,11.0,21.0,31.0,41.0,41.0,31.0,21.0,11.0,1.0,6.431380417335472,16.431380417335472,26.431380417335472,36.431380417335475,46.431380417335475"
OUTPUT_DIR="resources/simulation/acme"

for interval in {20..20..1}; do
    for i in {1..1}; do
        echo "Running simulation for interval $interval, replica $i"
        python -m bin.simulation.acme --change_interval $interval --core_update 3 --user_trajectory $TRAJECTORY --output_dir $OUTPUT_DIR
    done
done
