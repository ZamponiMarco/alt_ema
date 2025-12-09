#!/bin/bash
set -e

cd ..

# Configuration 1: 50 stations, 50 skew, 300 users, k=5, mu=(25,100)
python -m bin.model.gen_random_qn \
  --output-folder resources/random_qns_config1/ \
  --num-stations 50 \
  --num-networks 10 \
  --skewness 50 \
  --max-users 300 \
  --k-parameter 5 \
  --min-mu 25 \
  --max-mu 100 \
  --filter-target-max 0.6 \
  --filter-target-min 0.4

# Configuration 2: 30 stations, 30 skew, 180 users, k=4, mu=(15,60)
python -m bin.model.gen_random_qn \
  --output-folder resources/random_qns_config2/ \
  --num-stations 30 \
  --num-networks 10 \
  --skewness 30 \
  --max-users 180 \
  --k-parameter 4 \
  --min-mu 15 \
  --max-mu 60 \
  --filter-target-max 0.6 \
  --filter-target-min 0.4

# Configuration 3: 15 stations, 15 skew, 90 users, k=2, mu=(7.5,30)
python -m bin.model.gen_random_qn \
  --output-folder resources/random_qns_config3/ \
  --num-stations 15 \
  --num-networks 10 \
  --skewness 15 \
  --max-users 90 \
  --k-parameter 2 \
  --min-mu 7.5 \
  --max-mu 30 \
  --filter-target-max 0.6 \
  --filter-target-min 0.4

# Configuration 4: 10 stations, 10 skew, 60 users, k=2, mu=(5,20)
python -m bin.model.gen_random_qn \
  --output-folder resources/random_qns_config4/ \
  --num-stations 10 \
  --num-networks 10 \
  --skewness 10 \
  --max-users 60 \
  --k-parameter 2 \
  --min-mu 5 \
  --max-mu 20 \
  --filter-target-max 0.6 \
  --filter-target-min 0.4
