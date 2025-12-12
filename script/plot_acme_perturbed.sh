#!/bin/bash
set -e

cd ..

python -m bin.visualization.plot_load_acme_perturbed
python -m bin.visualization.plot_results_acme_perturbed