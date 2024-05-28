# learning_hierarchy
Public repository containing the code used in the hierarchical policy learning project. All data will be made publiclly available upon the acceptance of the manuscript.


## Citation
 ```
@article{li2024algorithmic,
title = {An algorithmic account for how humans efficiently learn, transfer, and compose hierarchically structured decision policies},
publisher = {PsyArXiv},
year = {2024},
doi = {https://doi.org/10.31234/osf.io/b3xnv},
url = {osf.io/preprints/psyarxiv/b3xnv},
author = {Jing-Jing Li and Anne G. E. Collins}
}
 ```


## File structure

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

### Plots
- `plots/`: folder storing plots generated by analysis code. 
