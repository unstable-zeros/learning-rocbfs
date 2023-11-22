# RO-CBF

##Setup the environment
```
conda create -n rocbf python=3.8.10
conda activate rocbf
pip install -r requirements.txt
```
## 

##Training Perception Map
```
python learning_cte.py
```
## Training Perception-based CBF
```
bash launch_image_data.sh
```
## Dependencies
* Carla 0.9.11 (we used overnight build version, but it should work for the full installation).
