# An algorithmic account for how humans efficiently learn, transfer, and compose hierarchically structured decision policies

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/pii/S0010027724002531">
    <img src="https://img.shields.io/badge/📝-Paper-green">
  </a>
</p>

**Authors:**
[Jing-Jing Li](https://jl3676.github.io/),
[Anne Collins](https://ccn.studentorg.berkeley.edu/)


## Task
A demo of the full online behavioral task used for data collection is available [here](https://experiments-ccn.berkeley.edu/learning_hierarchy_task_demo/exp.html?id=demo).

## File structure

### Data

This repository contains two versions of the data: `data.pkl` is the raw data used in the analyses presented in the paper and compatible with the code in this repository; `data.csv` contains the fully preprocessed data, which is more suitable for further analyses that do not depend on the functions in this repo. Both data files contain trial-by-trial stimulus, action, and reward information for all participants. 

`data.pkl`: raw pickled data file containing all behavioral data analyzed in the paper, compatible with the code in this repository (recommended for replications of analyses in the paper).

- `meta_data` contains the meta-data of each participant, including the experiment they participated in, the task version they completed, and the cluster they were assigned to. 
- `data` contains the data variables, organized first by participants: 
	- `s_12_12`: states (stimuli) in Blocks 1-2, Trials 1-60, Stages 1-2
	- `a_12_12`: actions in Blocks 1-2, Trials 1-60, Stages 1-2
	- `r_12_12`: rewards in Blocks 1-2, Trials 1-60, Stages 1-2
	- `counter12_12`: the number of key presses until reaching the correct action in Blocks 1-2, Trials 1-60, Stage 1-2
	- `s1`: states (stimuli) in Blocks 3-12, Trials 1-32, Stage 1
	- `s2`: states (stimuli) in Blocks 3-12, Trials 1-32, Stage 2
	- `a`: actions in Blocks 3-12, Trials 1-32, Stages 1-2
	- `r`: rewards in Blocks 3-12, Trials 1-32, Stages 1-2
	- `counter1`: the number of key presses until reaching the correct action in Blocks 3-12, Trials 1-32, Stage 1
	- `counter2`: the number of key presses until reaching the correct action in Blocks 3-12, Trials 1-32, Stage 2
	- `tr`: the randomized key mappings (K1-K4 to \[Q, W, E, R\] and K5-K8 to \[U, I, O, P\])

This file can be loaded using the following code: 
```python
import pickle

with open('data.pkl', 'rb') as file:
    all_data = pickle.load(file)
    meta_data = all_data['meta_data']
    data = all_data['data']
```

`data.csv`: contains fully preprocessed trial-by-trial behavioral data (recommended for further analyses that do not rely on the code in this repository). Each row represents a trial. The columns encode:

- `participant`: participant identifier
- `experiment`: experiment identifier (1=Experiment 1, 2=Experiment 2)
- `condition`: taks condition (V1-V1, V1-V2, V1-V3, V2-V1, V2-V2, V3-V1, V3-V3)
- `block`: block identifier (1-12)
- `trial`: trial identifier (1-60; numbering resets in each block)
- `s1`: stimulus identifier in stage 1 (1=gold, 2=silver)
- `a1`: list of actions selected in stage 1, separated by `;` (1=K1, 2=K2, 3=K3, 4=K4, 0=invalid key)
- `r1`: list of rewards in stage 1 corresponding to the actions, separated by `;` (1=reward, 0=no reward)
- `s2`: stimulus identifier in stage 2 (1=red, 2=blue)
- `a2`: list of actions selected in stage 2, separated by `;`  (5=K5, 6=K6, 7=K7, 8=K8, 0=invalid key)
- `r2`: list of rewards in stage 2 corresponding to the actions, separated by `;` (1=reward, 0=no reward)

Emply cell means no data (due to performance-based ). 


### Helper functions
- `helpers.py`: file containing helper functions for data analysis. 
- `plotting.py`: file containing plotting functions used to visualize results. 

### Data analysis
- `cluster_participants.ipynb`: notebook containing code to cluster participants based on their task performance.
- `analyze_behavior.ipynb`: notebook containing code to visualize the learning behavior of human participants. 

### Modeling
- `modeling.py`: file containing functions for generative cognitive modeling, model likelihood computation, and model fitting.
- `fit_models.ipynb`: notebook containing code to fit models to human behavior and visualize model validation. 
- `qc_modeling.ipynb`: notebook containing code for modeling quality control, such as parameter recovery. 

### Saved fitting results
- `fitting_results/`: folder containing saved model fitting results (likelihoods and parameter estimations), which can be loaded into `fit_models.ipynb` for analysis. 

## Citation

Please cite our work if you find this repository helpful!

 ```
@article{li2025algorithmic,
  title={An algorithmic account for how humans efficiently learn, transfer, and compose hierarchically structured decision policies},
  author={Li, Jing-Jing and Collins, Anne GE},
  journal={Cognition},
  volume={254},
  pages={105967},
  year={2025},
  publisher={Elsevier}
}
 ```
