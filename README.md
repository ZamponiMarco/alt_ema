# Automatic load test generation for elastic microservice applications: a falsification-based approach

## Installation

### Prerequisites

- Linux
- Python 3.12
- GUROBI Solver (license)
- Docker Engine

### Installation Steps

Clone the repository and create a virtual environment:

```
git clone https://github.com/ZamponiMarco/alt_ema
cd alt_ema

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install Python dependency [pycvxset](https://github.com/merlresearch/pycvxset)

Next, install Python dependencies:
```
pip install -r requirements.txt
```

## Running Experiments

All the experiments scripts are contained in the folder `script`, thus, from the project root go into the folder:

```
cd script
```

Here the steps to execute the three experiments are presented. After each experiment is concluded remove generated the data from the folder before starting a new one.

### AcmeAir

We start by generating the workload trace used for the test using the script `load_acme.sh`. The workload trace is saved in `resources/workloads/test.json`.

From the workload trace file, extract the trace and paste it in the `TRAJECTORY` field in the script `simulation_acme.sh` and then execute it, launching 10 30-minutes long test replays. After the execution, data extracted during the experiments will be saved in folder `resources/simulation`.

We can plot the results of the experiments calling the script `plot_acme.sh`. In this case, pictures plotting the experiment data can be found in folder `resources/pics`.

### Random

Start by generating the random QN models calling the script `generate_random.sh`. This will generate the folder `resources/random_qns` containing the list of models.

Consequently, generate the workloads for each qn solving the optimization problem by calling the script `load_random.sh`. This will generate the folder `resources/workloads` containing the optimal workload traces of each model.

Calling the script `count_random.sh` you obtain descriptive statistics about the generated workloads on the terminal, such as how many random models admit a falsifying workload trace and how many are instead infeasible for the falsification problem.

Finally, calling the script `simulation_random.sh` allows replaying the load tests on each random system. This will generate a `resources/simulation` folder containing measured performance metrics for each experiment.

Calling `plot_random.sh` the results of the experiments will be plotted in pictures stored in the folder `resources/pics`.

### Two-Tier

Calling the script `load_two_tier.sh` will generate a folder `resources/workloads` containing the workloads for all the combinations of elements (autoscaler/load shape/objective) in the optimization problem.

Consequently, calling the script `plot_two_tier.sh` will copy in the computer clipboard the LaTeX tables to report the results of the load generation.

## Paper Data

All the data from the experiments are contained in the folder `resources/paper_data`.