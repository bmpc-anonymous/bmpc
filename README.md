# Bootstrapped Model Predictive Control

----

*This repo is anonymized in compliance with ICLR 2025 submission guidelines.*

Implementation of Bootstrapped Model Predictive Control (BMPC).

----

## Installation

Install dependencies through conda.

```
conda env create -f docker/environment.yaml
pip install gym==0.21.0
```

Depending on your existing system packages, you may need to install other dependencies. See `docker/Dockerfile` for a list of recommended system packages.

## Training

See below examples on how to train an BMPC agent in the default setting. 

```
$ python train.py task=walker-walk steps=500000
$ python train.py task=dog-run steps=1000000
```

See `config.yaml` for a full list of arguments.
## Supported tasks

This codebase supports 28 tasks from DMControl, which covers all tasks used in the paper. See below table for expected name formatting:

| domain | task
| --- | --- |
| dmcontrol | walker-walk
| dmcontrol | finger-turn-easy
| dmcontrol | cartpole-balance-sparse
| dmcontrol | dog-run
| dmcontrol | dog-stand
| dmcontrol | humanoid-run

which can be run by specifying the `task` argument for `train.py`.

## HumanoidBench results

We further evaluate BMPC on [HumanoidBench](https://github.com/carlosferrazza/humanoid-bench). The corresponding evaluation performance is shown in the figure below. 
![image](./imgs/HumanoidBench%20result.png)
In the top left, we present the average performence of all tasks except for Reach due to the different reward scales. Mean and 95% CIs over 3 seeds.

## Reference

The code borrows heavily from nicklashansen's tdmpc2 [implementation](https://github.com/nicklashansen/tdmpc2).