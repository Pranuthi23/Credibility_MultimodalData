#!/bin/bash
n_trials=3
joint_training=True
group_tag="joint-training-$joint_training"
wandb=False
load_and_eval=True

for ((trial=1;trial<=n_trials;trial++)); 
do
    for dataset in "avmnist" "mmimdb"
    do
        python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        # python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial
    done
done
