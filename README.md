# Mirror, Mirror on the Wall---Have I Forgotten it All?\\A New Framework for Evaluating Machine Unlearning
Authors: Philip Mathew, Brennon Brimhall, Neil Fendley, Dr. Yinzhi Cao, Dr. Matthew D. Green

## Installation
After cloning, there are two major things that should be done.

1) Initialize the submodules for the unlearning frameworks
```bash
git submodule update --init --recursive
```

2) Install the uv environment
```bash
uv venv
uv sync
```

## Instructions
Experiments for the paper were run via the following command:
```bash
python swap_testing.py -c configs/gaussian_poison_experiment_config.yaml -o <INSERT OUTPUT FILE PATH>
```
The graphs and tables were then obtained from running through the `swap_vs_cu_graphs.ipynb` notebook. 
