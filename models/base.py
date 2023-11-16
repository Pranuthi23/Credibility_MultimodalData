from abc import ABC
from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision
from omegaconf import DictConfig
from rtpt import RTPT
from torch import nn
from models import *
# Translate the dataloader index to the dataset name
DATALOADER_ID_TO_SET_NAME = {0: "train", 1: "val", 2: "test"}

class LitModel(pl.LightningModule, ABC):
    """
    LightningModule for training a model using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration dictionary.
        name (str): Name of the model.
        steps_per_epoch (int): Number of steps per epoch.

    Attributes:
        cfg (DictConfig): Configuration dictionary.
        image_shape (ImageShape): Shape of the input data.
        rtpt (RTPT): RTPT logger.
        steps_per_epoch (int): Number of steps per epoch.
    """

    def __init__(self, cfg: DictConfig, name: str, steps_per_epoch: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.rtpt = RTPT(
            name_initials="SS",
            experiment_name="fusion_" + name + ("_" + str(cfg.tag) if cfg.tag else ""),
            max_iterations=cfg.epochs + 1,
        )
        self.save_hyperparameters()
        self.steps_per_epoch = steps_per_epoch


    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(0.7 * self.cfg.epochs), int(0.9 * self.cfg.epochs)],
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def on_train_start(self) -> None:
        self.rtpt.start()

    def on_train_epoch_end(self) -> None:
        self.rtpt.step()



class LateFusionDiscriminative(LitModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int):
        super().__init__(cfg, name="disc", steps_per_epoch=steps_per_epoch)

        self.encoders, self.predictors = [], []
        for modality in cfg.experiment.encoders:
            self.encoders += [eval(cfg.experiment.encoders[modality].type)(**cfg.experiment.encoders[modality].args)]
        for modality in cfg.experiment.predictors:
            self.predictors += [eval(cfg.experiment.predictors[modality].type)(**cfg.experiment.predictors[modality].args)]
        self.head = eval(cfg.experiment.head.type)(**cfg.experiment.head.args)
        self.encoders, self.predictors = torch.nn.ModuleList(self.encoders), torch.nn.ModuleList(self.predictors)
        self.num_modalities = cfg.experiment.dataset.modalities
        # Define loss function
        self.criterion = nn.NLLLoss()

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, on_step=True, prog_bar=True)
        self.log("Train/loss", loss, on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)

    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cross entropy loss and accuracy of batch.
        Args:
            batch: Batch of data.

        Returns:
            Tuple of (cross entropy loss, accuracy).
        """
        # Sanity check that there are #modalities + 1(target) variables in input
        assert len(batch) == self.num_modalities + 1
        
        data, labels = batch[:-1], batch[-1]
        predictions = []
        for unimodal_data, encoder, predictor in zip(data, self.encoders, self.predictors):
            predictions += [predictor(encoder(unimodal_data)).unsqueeze(0)]
        predictions = torch.cat(predictions, dim=0)
        predictions = self.head(predictions)
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions
        loss = self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy