#!/bin/bash

n_trials=1
wandb=True
load_and_eval=False
exp_name="neurips2024"

n_trials=1
wandb=False
load_and_eval=False
joint_training=True
group_tag="unbounded-dirichlet"
epochs=25
model="einet_dirichlet"
for ((trial=1;trial<=n_trials;trial++))
do  
    for dataset in  avmnist cub_mini nyud2 sunrgb_d 
    do
        python main.py dataset="$dataset" experiment="$dataset"_$model  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial epochs=$epochs 
        
    done
done

# for joint_training in "True" "False"
# do
#     group_tag="$exp_name/joint-training-$joint_training"
#     for ((trial=1;trial<=n_trials;trial++)); 
#     do
#             dataset=mmimdb
#             python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
            
#             dataset=mmimdb
#             python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial &
#             python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial 
            
#             # dataset=avmnist
#             # python main.py dataset="$dataset" experiment="$dataset"_mlp                                       load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial &
#             # python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=1 trial=$trial &
#             # python main.py dataset="$dataset" experiment="$dataset"_noisy_or                                  load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial &
#             # python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           load_and_eval=$load_and_eval joint_training=$joint_training group_tag=$group_tag wandb=$wandb gpu=0 trial=$trial 
            
           
#     done
# done

# n_trials=3
# group_tag='joint-training'

# # ## AVMNIST
# # dataset='avmnist'
# # for ((trial=1;trial<=n_trials;trial++)); 
# # do
# #     python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial 
# #     # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

# # done

# # ## MM-IMDB
# # dataset='mmimdb'
# # for ((trial=1;trial<=n_trials;trial++)); 
# # do
# #     python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
# #     python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial 
# #     # python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial 
# #     # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

# # done


# ## AVMNIST
# for ((trial=1;trial<=n_trials;trial++)); 
# do
#     dataset='avmnist'
#     python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial 
#     # python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial 
#     # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

# done

# ## MM-IMDB
# dataset='mmimdb'
# for ((trial=1;trial<=n_trials;trial++)); 
# do
#     python main.py dataset="$dataset" experiment="$dataset"_mlp                                       group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_weighted_mean                             group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_dirichlet                           group_tag=$group_tag wandb=True gpu=0 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_binomial                            group_tag=$group_tag wandb=True gpu=1 trial=$trial &
#     python main.py dataset="$dataset" experiment="$dataset"_einet_categorical                         group_tag=$group_tag wandb=True gpu=1 trial=$trial 
#     # python main.py dataset="$dataset" experiment="$dataset"_einet_conditional_categorical             group_tag=$group_tag wandb=True gpu=1 trial=$trial 
#     # python main.py dataset="$dataset" experiment="$dataset"_einet_multihead_conditional_categorical   group_tag=$group_tag wandb=True gpu=1 trial=$trial

# done