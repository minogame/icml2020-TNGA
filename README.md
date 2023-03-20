Usage:

First you need to start agents with

```CUDA_VISIBLE_DEVICES=0 python agent.py 0```

The last 0 stands for the id of the agent. 
You can spawn multiply agents with each one using one gpu by modifying the visible device id.

Then start the main script by

```python simulation_dis.py data.npz 50 40 200```

The argvs stands for the name of data, the numbers of individuals in one generation, the numbers of individuals after elimation, the alpha hyperparameter, correspondingly. 
