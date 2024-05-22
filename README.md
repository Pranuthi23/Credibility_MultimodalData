# Credibility_MultimodalData
This repository contains the code for the project - **Credibility-Aware Multi-Modal Fusion Using Probabilistic Circuits**. This is under active development.


## Setup
Create a new virtual environment and install the required packages given in `requirements.txt`.

**Submodule Dependencies**
This repository has dependencies with following three packages. They are organized in the `packages` directory.
- [MultiBench](https://github.com/pliang279/MultiBench)
- [RatSPN](https://github.com/braun-steven/spn-pytorch-experiments)

## To Run
Specify the hyperparameter configurations for your experiment in the appropriate config file inside `conf/`. 
Use the following commands to run experiments. You can pass values as needed from the command line for the hyperparameters specified in the config file.

```bash
python main.py dataset=avmnist experiment=avmnist_weighted_mean batch_size=128 group_tag=avmnist
```
```bash
python credibility.py dataset=avmnist experiment=avmnist_weighted_mean group_tag=avmnist lamda=0.1 noisy_modality=0
```

## Currently Supported Late Fusion Methods
- [x] Weighted Mean
- [x] Noisy-or
- [x] MLP
- [x] TMC
- [x] EinsumNet with Dirichlet leaves (Direct-PC)
- [x] Credibility Weighted Mean

    