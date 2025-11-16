# drl-final-project
Final project for Deep Reinforcement Learning

## Instructions

**Tested on Apple Silicon (M-series) with Python 3.10.**  
No MuJoCo required. we train/evaluate purely from fixed D4RL datasets.

### 1 Clone & set up the environment


git clone https://github.com/xiang233/drl-final-project.git
cd drl-final-project

```bash
conda env create -f env.yml
conda activate offline-ua
```

### 2 Get the offline datasets 

Download the HDF5 files from the D4RL mirror and place them under `~/.d4rl/`:
- `hopper-medium-replay-v2.hdf5`
- `walker2d-medium-v2.hdf5`

```bash
mkdir -p ~/.d4rl
```



### 3 Run the Behavior Cloning baseline

```bash
python -m scripts.train_bc --env hopper-medium-replay-v2 --steps 20000 --seed 0

python -m scripts.train_bc --env walker2d-medium-v2 --steps 20000 --seed 0
```
