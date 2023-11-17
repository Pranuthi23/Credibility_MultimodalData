#!/bin/bash
python main.py dataset=avmnist experiment=avmnist_mlp group_tag=detached_head wandb=True
python main.py dataset=avmnist experiment=avmnist_weighted_mean group_tag=detached_head wandb=True
python main.py dataset=avmnist experiment=avmnist_einet_dirichlet group_tag=detached_head wandb=True
python main.py dataset=avmnist experiment=avmnist_einet_binomial group_tag=detached_head wandb=True
