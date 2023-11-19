#!/bin/bash
n_trials=5
group_tag='no-gradient-from-head'

## dataset='avmnist'
# for ((trial=1;trial<=n_trials;trial++)); 
# do
#     python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

# done

## MM-IMDB
dataset='mmimdb'
for ((trial=1;trial<=n_trials;trial++)); 
do
    python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
    python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial 
    # python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial 
    # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

done