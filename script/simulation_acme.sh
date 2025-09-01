#!/bin/bash
set -e

cd ..

TRAJECTORY="10.0,10.0,10.0,10.0,1.931380417334065,11.9313804173369,21.93138041733692,31.93138041733696,41.931380417337,51.93138041733706,60.0,60.0,50.0,40.0,30.0,20.0,15.477272727271929,5.4772727272741,1.0,1.0,1.0,1.0,6.954253611571359,16.954253611571396,26.954253611571417,36.95425361157146,46.95425361157149,56.95425361157149,60.0,60.0"
OUTPUT_DIR="resources/simulation/acme"

for interval in {60..60..1}; do
    for i in {1..10}; do
        echo "Running simulation for interval $interval, replica $i"
        python -m bin.simulation.acme --change_interval $interval --core_update 3 --user_trajectory $TRAJECTORY --output_dir $OUTPUT_DIR
    done
done
