#!/bin/bash
n_trials=5
group_tag='no-gradient-from-head'

for ((trial=1;trial<=n_trials;trial++)); 
do
    python main.py dataset=avmnist experiment=avmnist_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial &
    python main.py dataset=avmnist experiment=avmnist_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

done
