# Mirror, Mirror on the Wall---Have I Forgotten it All?\\A New Framework for Evaluating Machine Unlearning
Authors: Philip Mathew, Brennon Brimhall, Neil Fendley, Dr. Yinzhi Cao, Dr. Matthew D. Green

## Installation
After cloning, there are two major things that should be done.

1) Initialize the submodules for the unlearning frameworks
```bash
cd unlearning_frameworks/
git submodule init
git submodule update --remote --merge
git clone https://github.com/if-loops/selective-synaptic-dampening.git
mv selective-synaptic-dampening selective_synaptic_dampening
```

2) Install the conda environment
```bash
conda env create -f environment.yaml
conda activate unlearning
```

## Instructions
Experiments can be run via the `run_experiment.py` script, which takes in a YAML file as a config. Read through experiment_config.yaml for an idea of what the config YAML file looks like.
