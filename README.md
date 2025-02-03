# Performance Dependent Network Evolution (PDNE)

- This repository contains codes for simulating the network evolution framework provided in the manuscript **''Evolution Beats Random Chance: Performance-dependent Network Evolution for Enhanced Computational Capacity''**. 
- Authors: Manish Yadav, Sudeshna Sinha and Merten Stender
- Article: https://journals.aps.org/pre/abstract/10.1103/PhysRevE.111.014320
  
<p align="center">
<img src="https://github.com/maneesh51/PDNE/blob/main/Fig1.png">
</p>


## 1. Brief description of the framework
The fundamental problem of identifying *structure-function* relationships in networks has gained significant attention across diverse scientific fields in recent years. We provide a new perspective on the same problem by taking a step back and asking how networks evolve to achieve specific functionality. What are the underlying principles and scaling laws? Is it possible to generate optimally sized networks to solve specific tasks? 

We addressed the aforementioned problem by proposing a performance-dependent network evolution framework. The framework consists of node addition and deletion modules that selectively expand and shrink the underlying network to improve the overall information processing capabilities of the network. The unique yet simple idea of a goal-oriented network evolution framework has helped us to elucidate the emergent *structure-function* relationships via graph-theoretic measures and task-specific networks of distinct size and densities with their unique organizations, growth rates and internal asymmetries. For more details: https://arxiv.org/abs/2403.15869

## 2. Description of the files present in this repository
All the functions required to build the PDNE framework and also to reproduce the results in the manuscript are present in `PDNE_Functions.py`. All the simulations can be run using the Python script `PDNE_RunTasks.py` and also using the Python notebook `PDNE_RunTasks.ipynb`. All the figures can be regenerated using `PDNE_Plots_A.ipynb` and `PDNE_Plots_B.ipynb` notebooks.

## 3. Data
Data required to reproduce the figures in the manuscript can be downloaded using the following [Dropbox link](https://www.dropbox.com/scl/fo/l5f6jyu4smrg7ytk4xs1u/AMfVw_V5PcRnMCV2UfZ_MOE?rlkey=q28e7bce9fd6698bvxywj9i9y&st=ftnmaqnr&dl=0)

## 4. Online executable code
An online executable version of the code is also provided in the Python notebook format on [Google Colab](https://colab.research.google.com/drive/10z6Bs2C83DwtnomQdFChYKzzjCUIL0xi?usp=sharing)


## 5. Specific Python libraries and their versions
- **NetworkX:** version 2.8.4 (https://networkx.org/documentation/stable/release/release_2.8.4.html)
- **Matplotlib:** version 3.8.0 (https://matplotlib.org/stable/users/prev_whats_new/whats_new_3.8.0.html)
- **Scipy:** version 1.10.1 (https://docs.scipy.org/doc/scipy-1.10.1/)
