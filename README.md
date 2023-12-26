# Credibility_MultimodalData
This repository contains the code for the project - **Probabilistic Multi-Modal Discriminative Learning**. This is under active development.

## Experimental Results
Current experimental results and observations for this project are summarized here [here](https://api.wandb.ai/links/sahil-sidheekh/x00203ky).

## Setup
Create a new virtual environment and install the required packages given in `requirements.txt`.

**Submodule Dependencies**
This repository has dependencies with following three packages. They are organized in the `packages` directory.
- [MultiBench](https://github.com/braun-steven/spn-pytorch-experiments)
- [RatSPN](https://github.com/pliang279/MultiBench)

## To Run
Specify the hyperparameter configurations for your experiment in the appropriate config file inside `conf/`. 
Use the following command to run experiments. Ypou can pass values as needed from the command line for the hyperparameters specified in the config file.

```bash
python main.py dataset=avmnist experiment=avmnist_weighted_mean batch_size=128
```

## Currently Supported and under development Late Fusion Methods
- [x] Weighted Mean
- [x] RatSPN with dirichlet leaves. This does not seem to be learning - **needs debugging**. 
- [x] EinsumNet with Dirichlet and Binomial leaves
- [x] Try Categorical leaf with gumbel softmax
- [ ] Try using with Gaussians in feature space - EarlyFusion
- [ ] Implement joint distribution over y, y_i for PC 

The current training methodology involves independent training of combination function and unimodal predictors.
Joint training seems unstable for PCs. 

- [ ] Try using warmstart, independent training and then fine tuning

    