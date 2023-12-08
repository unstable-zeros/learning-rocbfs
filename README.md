# RO-CBF

This repository contains the code needed to reproduce the examples found in 
L. Lindemann, A. Robey, L. Jiang, S. Das, S. Tu, N. Matni, [Learning Robust Output Control Barrier Functions from Safe Expert Demonstrations](https://arxiv.org/abs/2111.09971)

## Setup the environment
```
conda create -n rocbf python=3.8.10
conda activate rocbf
pip install -r requirements.txt
```
## 

## Training Perception Map
```
python learning_cte.py
```
## Training Perception-based ROCBF
```
bash launch_image_data.sh
```
## Dependencies
* Carla 0.9.11 (we used overnight build version, but it should work for the full installation).
