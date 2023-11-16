# Credibility_MultimodalData
This repository contains the code for the project - **Probabilistic Multi-Modal Discriminative Learning**

## Setup
Create a new virtual environment and install the required packages given in `requirements.txt`.

- Submodule Dependencies

## To Run
Specify the hyperparameter configurations for your experiment in the appropriate config file inside `conf/`. 
Use the following command to run experiments. Ypou can pass values as needed from the command line for the hyperparameters specified in the config file.

```bash
python main.py dataset=avmnist experiment=avmnist_weighted_mean batch_size=128
```