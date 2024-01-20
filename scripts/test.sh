#!/bin/bash

n_trials=3
wandb=True
load_and_eval=True
joint_training=True

group_tag="joint-training-$joint_training"
for ((trial=1;trial<=n_trials;trial++))
do
        dataset=mmimdb
        python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        
        dataset=avmnist
        python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
             
done