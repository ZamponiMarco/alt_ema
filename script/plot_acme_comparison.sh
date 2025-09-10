#!/bin/bash
set -e

cd ..

python -m bin.visualization.plot_comparison_samples
python -m bin.visualization.plot_comparison_timings