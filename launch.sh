#!/bin/bash

export SYSTEM='carla'
export ROOT='./data/carla/left_turn_state_space_sampling'
export FNAME='Data_Collection_Compiled_sample_state_space'

# middle of turn
export DATA_PATH_1="${ROOT}/middle_of_turn/gridding/${FNAME}_5.pd"
export DATA_PATH_2="${ROOT}/middle_of_turn/random_sampling/${FNAME}_9.pd"

# quarter of turn
export DATA_PATH_3="${ROOT}/quarter_of_turn/gridding/${FNAME}_4.pd"

# start of turn
export DATA_PATH_4="${ROOT}/Start_of_Turn/gridding/${FNAME}_3.pd"
export DATA_PATH_5="${ROOT}/Start_of_Turn/random_sampling/${FNAME}_7.pd"

# straight line
export DATA_PATH_6="${ROOT}/Straight_Lane_Driving/gridding/${FNAME}_1.pd"
export DATA_PATH_7="${ROOT}/Straight_Lane_Driving/gridding/${FNAME}_2.pd"
export DATA_PATH_8="${ROOT}/Straight_Lane_Driving/random_sampling/${FNAME}_8.pd"

# third quarter turn
export DATA_PATH_9="${ROOT}/third_quarter_turn/gridding/${FNAME}_6.pd"

export RIGHT_ROOT='./data/carla/right_turn_state_space_sampling'
export RIGHT_FNAME='Data_Collection_Compiled_reverse_sample_state_space'

# (right) quarter of turn
export DATA_PATH_10="${RIGHT_ROOT}/quarter_of_turn/gridding/${RIGHT_FNAME}_2.pd"

# (right) start of turn
export DATA_PATH_11="${RIGHT_ROOT}/start_of_turn/gridding/${RIGHT_FNAME}_1.pd"

export RESULTS_PATH='./results-tmp'

# Margins for optimization constraints 
export GAMMA_SAFE=0.3
export GAMMA_UNSAFE=0.3
export GAMMA_DYN=0.01

# Lagrange multipliers (fixed)
export LAMBDA_GRAD=0.01
export LAMBDA_PARAM=0.01

# Robustness
export DELTA_F=0.5
export DELTA_G=0.5

# Output map
export LIP_CONST_1=0.2
export LIP_CONST_2=0.2

# Training
export NET_DIMS=(32 16)
export N_EPOCHS=2000
export LEARNING_RATE=0.005
export DUAL_STEP_SIZE=0.05
export DUAL_SCHEME='ae'

# For clean data with data aug
export NEIGHBOR_THRESH=0.008
export MIN_N_NEIGHBORS=200

# Additional state sampling
export N_SAMP_UNSAFE=0
export N_SAMP_SAFE=0
export N_SAMP_ALL=0




export CUDA_VISIBLE_DEVICES=0
python main.py \
    --system $SYSTEM --data-path $DATA_PATH_1 $DATA_PATH_2 $DATA_PATH_3 $DATA_PATH_4 \
    $DATA_PATH_5 $DATA_PATH_6 $DATA_PATH_7 $DATA_PATH_8 $DATA_PATH_9 $DATA_PATH_10 $DATA_PATH_11 \
    --results-path $RESULTS_PATH \
    --gamma-safe $GAMMA_SAFE --gamma-unsafe $GAMMA_UNSAFE --gamma-dyn $GAMMA_DYN \
    --lambda-grad $LAMBDA_GRAD --lambda-param $LAMBDA_PARAM \
    --net-dims ${NET_DIMS[@]} --n-epochs $N_EPOCHS \
    --learning-rate $LEARNING_RATE --dual-step-size $DUAL_STEP_SIZE \
    --nbr-thresh $NEIGHBOR_THRESH --min-n-nbrs $MIN_N_NEIGHBORS \
    --n-samp-unsafe $N_SAMP_UNSAFE --n-samp-safe $N_SAMP_SAFE --n-samp-all $N_SAMP_ALL \
    --dual-scheme $DUAL_SCHEME \
    --robust --delta-f $DELTA_F --delta-g $DELTA_G \
    --use-lip-output-term --lip-const-a $LIP_CONST_1 --lip-const-b $LIP_CONST_2
    
