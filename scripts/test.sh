#!/bin/bash

# n_trials=3
# wandb=True
# load_and_eval=True
# joint_training=True

# group_tag="joint-training-$joint_training"
# for ((trial=1;trial<=n_trials;trial++))
# do
#         # dataset=mmimdb
#         # python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
#         # python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
#         # python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
#         # python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
        
#         dataset=avmnist
#         python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
#         python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
#         python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
#         python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
             
# done

n_trials=3
wandb=False
load_and_eval=True
joint_training=True

group_tag="UAI"
for ((trial=1;trial<=n_trials;trial++))
do
        
        dataset=avmnist
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_einet_dirichlet_T"$trial.txt 
        python main.py dataset="$dataset" experiment="$dataset"_credibility_weighted           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_credibility_weighted_T"$trial.txt

        dataset=cub_mini
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_einet_dirichlet_T"$trial.txt 
        python main.py dataset="$dataset" experiment="$dataset"_credibility_weighted           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_credibility_weighted_T"$trial.txt

        dataset=nyud2
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_einet_dirichlet_T"$trial.txt 
        python main.py dataset="$dataset" experiment="$dataset"_credibility_weighted           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_credibility_weighted_T"$trial.txt

        dataset=sunrgb_d
        python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_einet_dirichlet_T"$trial.txt 
        python main.py dataset="$dataset" experiment="$dataset"_credibility_weighted           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial > console/$dataset"_credibility_weighted_T"$trial.txt
                     
done
