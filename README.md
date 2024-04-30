# Machine Unlearning is Computational Indistinguishability
Authors: Philip Mathew, Brennon Brimhall, Neil Fendley

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
The three main scripts are `generate_forget_set.py`, `train_m1_m3.py`, and `unlearn_forget_set.py`.
- `generate_forget_set.py`: Creates a CSV of indices corresponding to samples that should be forgotten. Supports both randomly-generated forget sets and class-specific forget sets
- `train_m1_m3.py`: Simple training script to train one model $M_1$ from the full dataset and another model $M_3$ from only the retain set
- `unlearn_forget_set.py`: Generates unlearned model $M_2$ from $M_1$ and an associated forget set. 

For all 3 scripts, simply typing `python <SCRIPT_PATH> --help` will give you a list of all of its possible command line arguments.
