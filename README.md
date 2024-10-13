markdown: kramdown

# learning_hierarchy
Public repository containing the code used in the hierarchical policy learning project. All data will be made publiclly available upon the acceptance of the manuscript.


## Citation
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

## Task
A demo of the full online behavioral task used for data collection is available [here](https://experiments-ccn.berkeley.edu/learning_hierarchy_task_demo/exp.html?id=demo).

## File structure

### Data
- `data.pkl`: pickle file containing all behavioral data analyzed in the paper.
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
