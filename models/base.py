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
import torchmetrics
import wandb

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
        self.configure_metrics()
    
    def configure_metrics(self):
       return "Not Implemented"
   
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



class FusionModel(LitModel):
    """
    Fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusion"):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.cfg = cfg
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
        loss, accuracy, predictions = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, on_step=True, prog_bar=True)
        self.log("Train/loss", loss, on_step=True)
        self.log_metrics(predictions, train_batch[-1], "Train/", on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy, predictions = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        self.log_metrics(predictions, val_batch[-1], "Val/")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy, predictions = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)
        self.log_metrics(predictions, batch[-1], f"Test/{set_name}_", add_dataloader_idx=False)
    
    def log_metrics(self, probs, targets, mode='Train/', **kwargs):
        probs, targets = probs.detach().cpu(), targets.detach().cpu()
        for metric_name in self.metrics:
            self.log(f"{mode}{metric_name}",self.metrics[metric_name](probs, targets),**kwargs)
        
    def _get_cross_entropy_and_accuracy(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    

class LateFusionClassifier(FusionModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusionClassifier", ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)

    def configure_metrics(self):
        self.metrics = {
            'AUROC': torchmetrics.AUROC(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multiclass", average='macro', num_classes=self.cfg.experiment.dataset.num_classes),
        }

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
        loss, embeddings, predictions = 0.0,  [], []
        for unimodal_data, encoder, predictor in zip(data, self.encoders, self.predictors):
            embeddings += [encoder(unimodal_data)]
            unimodal_prediction = predictor(embeddings[-1])
            if(self.cfg.experiment.head.threshold_input):
                predictions += [unimodal_prediction.argmax(dim=-1).unsqueeze(1)]
            else:
                predictions += [unimodal_prediction.unsqueeze(1)]
            ll_y_g_x = unimodal_prediction.log()
            loss += self.criterion(ll_y_g_x, labels)
        predictions = torch.cat(predictions, dim=1)
        if(not self.cfg.joint_training):
            predictions = predictions.detach()
        predictions = self.head(predictions, embeddings)
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions.log()
        loss += self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions
    


class LateFusionMultiLabelClassifier(FusionModel):
    """
    Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="LateFusionMultiLabelClassifier", ):
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.criterion = nn.BCELoss()
        self.accuracy = torchmetrics.Accuracy(task="multilabel", num_labels= self.cfg.experiment.dataset.num_classes)
        
    def configure_metrics(self):
        self.metrics = {
            'AUROC': torchmetrics.AUROC(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Precision': torchmetrics.Precision(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Recall': torchmetrics.Recall(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'F1Score': torchmetrics.F1Score(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
            'Accuracy': torchmetrics.Accuracy(task="multilabel", average='weighted', num_labels= self.cfg.experiment.dataset.num_classes),
        }

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
        loss, embeddings, predictions = 0.0,  [], []
        
        for unimodal_data, encoder, predictor in zip(data, self.encoders, self.predictors):
            embeddings += [encoder(unimodal_data)]
            unimodal_prediction = predictor(embeddings[-1])
            predictions += [unimodal_prediction.unsqueeze(1)]
            loss += self.criterion(unimodal_prediction, labels.to(unimodal_prediction.dtype))
        
        predictions = torch.cat(predictions, dim=1)
        if(not self.cfg.joint_training):
            predictions = predictions.detach()
        predictions = torch.cat([predictions.unsqueeze(-1),1-predictions.unsqueeze(-1)], dim=-1)
        
        if(self.cfg.experiment.head.threshold_input):
            predictions = predictions.argmax(dim=-1)
        
        predictions = self.head(predictions, embeddings)
        loss += self.criterion(predictions.to(torch.float64), labels.to(torch.float64))
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy, predictions
    


class EarlyFusionDiscriminative(FusionModel):
    """
    Early fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """

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
        loss, embeddings = 0.0,  []
        for unimodal_data, encoder in zip(data, self.encoders):
            embeddings += [encoder(unimodal_data)]
        embeddings = torch.cat(embeddings, dim=-1)
        predictions = self.head(embeddings)
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions.log()
        loss += self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions