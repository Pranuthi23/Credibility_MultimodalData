# Credibility_MultimodalData
This repository contains the code for the project - **Probabilistic Multi-Modal Discriminative Learning**. This is under active development.

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

## Currently Supported Late Fusion Methods
- Weighted Mean
- RatSPN with dirichlet leaves. This does not seem to be learning - **needs debugging**. Next steps
    - [ ] Try Replacing with Gaussians
    - [ ] Implement joint distribution over y, y_i for PC 
    - [ ] Try Categorical leaf with gumbel softmax
    