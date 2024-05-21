#!/usr/bin/env python
import os 
import sys
from packages import PACKAGE_DICT 
for package in PACKAGE_DICT:
    print(f"Adding package: {package} to sys.path. Given path: {os.path.join('packages', PACKAGE_DICT[package])}")
    sys.path.append(os.path.join("packages", PACKAGE_DICT[package]))
import omegaconf
import time
import wandb
from hydra.core.hydra_config import HydraConfig
import logging
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from rich.traceback import install
install()
import hydra
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import (
    ModelSummary,
)
from utils import (
    load_from_checkpoint,
)

from datasets import get_dataloader
from models.base import LateFusionClassifier, LateFusionMultiLabelClassifier, FusionModel
from models import *
import json 

# A logger for this file
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

 
class JSD(torch.nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = torch.nn.KLDivLoss(reduction='none', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor, eps=1e-12):
        p, q = p.clamp(eps), q.clamp(eps)
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(p.log(), m) + self.kl(q.log(), m))
    
class NoisyLateFusionClassifier(LateFusionClassifier):
    """
    Noisy Late fusion model. Outputs the probability distribution over a target variable given input from multiple modalities
    """
    def __init__(self, cfg: DictConfig, steps_per_epoch: int, name="NoisyLateFusionClassifier"):
        self.lamda = cfg.lamda 
        super().__init__(cfg, name=name, steps_per_epoch=steps_per_epoch)
        self.alpha = torch.sigmoid(torch.zeros(cfg.experiment.dataset.num_classes))
        self.test_credibility = []
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
        for i, (unimodal_data, encoder, predictor) in enumerate(zip(data, self.encoders, self.predictors)):
            if(encoder is not None):
                embeddings += [encoder(unimodal_data)]
            else:
                embeddings += [unimodal_data]
            unimodal_prediction = predictor(embeddings[-1])
            if(i==self.cfg.experiment.noisy_modality):
                noise = torch.distributions.Dirichlet(self.alpha).sample([unimodal_data.shape[0]]).to(unimodal_prediction.device)
                unimodal_prediction = self.lamda*unimodal_prediction + (1-self.lamda)*noise
            if(self.cfg.experiment.head.threshold_input):
                predictions += [unimodal_prediction.argmax(dim=-1).unsqueeze(1)]
            else:
                predictions += [unimodal_prediction.unsqueeze(1)]
                
        predictions_in = torch.cat(predictions, dim=1).detach()
        predictions_out = self.head(predictions_in)
        # Criterion is NLL which takes logp( y | x)
        # NOTE: Don't use nn.CrossEntropyLoss because it expects unnormalized logits
        # and applies LogSoftmax first
        ll_y_g_x = predictions_out.log()
        loss = self.criterion(ll_y_g_x, labels)
        accuracy = (labels == ll_y_g_x.argmax(-1)).sum() / ll_y_g_x.shape[0]
        return loss, accuracy, predictions_in, predictions_out
    
    def training_step(self, train_batch, batch_idx):
        loss, accuracy, predictions_in, predictions_out = self._get_cross_entropy_and_accuracy(train_batch)
        self.log("Train/accuracy", accuracy, on_step=True, prog_bar=True)
        self.log("Train/loss", loss, on_step=True)
        self.log_metrics(predictions_in, predictions_out, train_batch[-1], "Train/", on_step=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy, predictions_in, predictions_out = self._get_cross_entropy_and_accuracy(val_batch)
        self.log("Val/accuracy", accuracy, prog_bar=True)
        self.log("Val/loss", loss)
        self.log_metrics(predictions_in, predictions_out, val_batch[-1], "Val/")
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, accuracy, predictions_in, predictions_out = self._get_cross_entropy_and_accuracy(batch)
        set_name = DATALOADER_ID_TO_SET_NAME[dataloader_idx]
        if(set_name == "test"):
            self.test_pred += [predictions_out]
            self.test_target += [batch[-1]]
        self.log(f"Test/{set_name}_accuracy", accuracy, add_dataloader_idx=False)
        self.log_metrics(predictions_in, predictions_out, batch[-1], f"Test/{set_name}_", add_dataloader_idx=False)
    
    def log_metrics(self, predictions_in, predictions_out, targets, mode='Train/', **kwargs):
        credibility = []
        for i in range(self.cfg.experiment.dataset.modalities):
            p_y_pi = self.head(predictions_in, marginalized_scopes=[i]).exp()
            p_y_pi = p_y_pi / p_y_pi.sum(dim=-1, keepdim=True)
            credibility += [-torch.nn.functional.kl_div(predictions_out,p_y_pi).view(-1,1)]
            # credibility += [-JSD()(predictions_out,p_y_pi.exp()).exp().sum(dim=-1).view(-1,1)]
        credibility = torch.cat(credibility,dim=-1)
        credibility = credibility/credibility.sum(dim=-1, keepdim=True)
        credibility = credibility.mean(dim=0).detach().cpu()
        for i in range(self.cfg.experiment.dataset.modalities):     
            self.log(f"{mode}Credibility-Modality-{i}",credibility[i])
        probs, targets = predictions_out.detach().cpu(), targets.detach().cpu()
        for metric_name in self.metrics:
            self.log(f"{mode}{metric_name}",self.metrics[metric_name](probs, targets),**kwargs)
    
    


def main(cfg: DictConfig):
    """
    Main function for training and evaluating an Einet.

    Args:
        cfg: Config file.
    """
    preprocess_cfg(cfg)

    # Get hydra config
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir
    logger.info("Working directory : {}".format(os.getcwd()))

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # Safe run_dir in config (use open_dict to make config writable)
    with open_dict(cfg):
        cfg.run_dir = run_dir

    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    logger.info("Run dir: " + run_dir)

    seed_everything(cfg.seed, workers=True)

    if not cfg.wandb:
        os.environ["WANDB_MODE"] = "offline"

    # Ensure that everything is properly seeded
    seed_everything(cfg.seed, workers=True)

    # Setup devices
    if torch.cuda.is_available():
        accelerator = "gpu"
        if type(cfg.gpu) == int:
            devices = [int(cfg.gpu)]
        else:
            devices = [int(g) for g in cfg.gpu]
    else:
        accelerator = "cpu"
        devices = 1

    # Create dataloader
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    # Create callbacks
    cfg_container = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger_wandb = WandbLogger(
        name=cfg.tag,
        project=cfg.project_name,
        group=cfg.group_tag,
        offline=not cfg.wandb,
        config=cfg_container,
        reinit=True,
        save_dir=run_dir,
        settings=wandb.Settings(start_method="thread"),
    )

    # Load or create model
    base_model_class = LateFusionMultiLabelClassifier if(cfg.experiment.multilabel) else LateFusionClassifier
    noisy_model_class = None if(cfg.experiment.multilabel) else NoisyLateFusionClassifier
    base_model = base_model_class.load_from_checkpoint(f"{run_dir}/best_model.pt")
    noisy_model = noisy_model_class(cfg, steps_per_epoch=len(train_loader))
    noisy_model.load_state_dict(base_model.state_dict())
    
    # Setup callbacks
    ckpt_callback = ModelCheckpoint(f"{run_dir}/checkpoints", monitor="Val/F1Score", save_top_k=3, mode='max')
    callbacks = [
        ckpt_callback
    ]
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger_wandb,
        accelerator=accelerator,
        devices=devices,
        callbacks=callbacks,
        precision=cfg.precision,
        fast_dev_run=cfg.debug,
        profiler=cfg.profiler,
        default_root_dir=run_dir,
        enable_checkpointing=True,
        detect_anomaly=True,
    )

    # if not cfg.load_and_eval:
    #     # Fit model
    #     trainer.fit(model=noisy_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    #     noisy_model = noisy_model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 

    performance_summary = {
        "no_noise": {},
        "noise_in_test": {},
        "noise_in_test_and_train":{}
    }
    
    logger.info("Evaluating Orignal Base Model...")
    trainer.test(model=base_model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)
    
            
    logger.info("Evaluating Noisy Trained Model...")
    trainer.test(model=noisy_model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)
    
    for metric_name in base_model.metrics:
        metric = base_model.metrics[metric_name].cpu()
        performance_summary["no_noise"][metric_name] = metric(base_model.test_pred, base_model.test_target).item()
        
    for metric_name in noisy_model.metrics:
        metric = noisy_model.metrics[metric_name].cpu()
        performance_summary["noise_in_test"][metric_name] = metric(noisy_model.test_pred, noisy_model.test_target).item()
        
    # trainer.fit(model=noisy_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # noisy_model = noisy_model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 
    
    # trainer.test(model=noisy_model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)
    # for metric_name in noisy_model.metrics:
    #     metric = noisy_model.metrics[metric_name].cpu()
    #     performance_summary["noise_in_test_and_train"][metric_name] = metric(noisy_model.test_pred, noisy_model.test_target).item()
    
    # mean_credibility = []
    # for batch in test_loader:
    #     loss, accuracy, predictions_in, predictions_out = noisy_model._get_cross_entropy_and_accuracy(batch)
    #     credibility = []
    #     for i in range(noisy_model.cfg.experiment.dataset.modalities):
    #         p_y_pi = noisy_model.head(predictions_in, marginalized_scopes=[i]).exp()
    #         p_y_pi = p_y_pi / p_y_pi.sum(dim=-1, keepdim=True)
    #         credibility += [-torch.nn.functional.kl_div(predictions_out,p_y_pi).view(-1,1)]
    #         # credibility += [-JSD()(predictions_out,p_y_pi.exp()).exp().sum(dim=-1).view(-1,1)]
    #     credibility = torch.cat(credibility,dim=-1)
    #     credibility = credibility/credibility.sum(dim=-1, keepdim=True)
    #     credibility = credibility.mean(dim=0).detach().cpu()
    #     mean_credibility += [credibility.view(1,-1)]
    # mean_credibility = torch.cat(mean_credibility, dim=0).mean(dim=0)
    # performance_summary["noise_in_test_and_train"]["mean_credibility"] = mean_credibility.detach().cpu().numpy()
    
    print(performance_summary)
    summary_dir = os.path.join(run_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir,f'noisy-modality={cfg.experiment.noisy_modality}-lamda={cfg.lamda}.json'), 'w') as fp:
        json.dump(performance_summary, fp)
    # # Save checkpoint in general models directory to be used across experiments
    # chpt_path = os.path.join(run_dir, "best_noisy_model.pt")
    # logger.info("Saving checkpoint: " + chpt_path)
    # trainer.save_checkpoint(chpt_path)
    # device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    # # Create dataloader
    # train_loader, val_loader, test_loader = get_dataloader(cfg)
    # alpha = torch.sigmoid(torch.zeros(cfg.experiment.dataset.num_classes))
    # for lamda in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
    #     model = LateFusionClassifier.load_from_checkpoint(f"{run_dir}/best_model.pt").to(device)
    #     optimizer = torch.optim.Adam(model.head.parameters())
    #     for epoch in range(10):
    #         for batch in train_loader:
    #             data, labels = batch[:-1], batch[-1].to(device)
    #             loss, embeddings, predictions = 0.0,  [], []
    #             for i, (unimodal_data, encoder, predictor) in enumerate(zip(data, model.encoders, model.predictors)):
    #                 embeddings += [encoder(unimodal_data.to(device).to(torch.float32))]
    #                 unimodal_prediction = predictor(embeddings[-1])
    #                 noise = torch.distributions.Dirichlet(alpha).sample([unimodal_data.shape[0]]).to(device)
    #                 if(i==0):
    #                     unimodal_prediction = lamda*unimodal_prediction + (1-lamda)*noise
    #                 if(model.cfg.experiment.head.threshold_input):
    #                     predictions += [unimodal_prediction.argmax(dim=-1).unsqueeze(1)]
    #                 else:
    #                     predictions += [unimodal_prediction.unsqueeze(1)]
    #             predictions = torch.cat(predictions, dim=1)
    #             predictions = predictions.detach()
    #             p_y_p1_p2 = model.head.model(predictions.unsqueeze(1).unsqueeze(3).unsqueeze(3)).exp()
    #             p_y_p1_p2 = p_y_p1_p2/p_y_p1_p2.sum(dim=-1, keepdim=True)
    #             ll_y_g_x = p_y_p1_p2.log()
    #             loss += model.criterion(ll_y_g_x, labels)
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             credibility, marginal_probs = [], []
    #             for i in range(cfg.experiment.dataset.modalities):
    #                 p_y_pi = model.head.model(predictions.unsqueeze(1).unsqueeze(3).unsqueeze(3), marginalized_scopes=[i]).exp()
    #                 marginal_probs += [p_y_pi/p_y_pi.sum(dim=-1, keepdim=True)]
    #                 credibility += [-torch.nn.functional.kl_div(p_y_p1_p2,p_y_pi).view(-1,1)]
    #             print(f"ep-{epoch}",f"Loss:{loss.item()}", torch.cat(credibility,dim=1).mean(dim=0))
                
def preprocess_cfg(cfg: DictConfig):
    """
    Preprocesses the config file.
    Replace defaults if not set (such as data/results dir).

    Args:
        cfg: Config file.
    """
    home = os.getenv("HOME")
    
    # If FP16/FP32 is given, convert to int (else it's "bf16", keep string)
    if cfg.precision == "16" or cfg.precision == "32":
        cfg.precision = int(cfg.precision)

    if "profiler" not in cfg:
        cfg.profiler = None  # Accepted by PyTorch Lightning Trainer class

    if "tag" not in cfg:
        cfg.tag = cfg.experiment.name+"-noisy"

    # cfg.group_tag = "credibility-analysis"

    if "seed" not in cfg:
        cfg.seed = int(time.time())
        

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main_hydra(cfg: DictConfig):
    try:
        main(cfg)
    except Exception as e:
        logging.critical(e, exc_info=True)  # log exception info at CRITICAL log level
    finally:
        # Close wandb instance. Necessary for hydra multi-runs where main() is called multipel times
        wandb.finish()


if __name__ == "__main__":
    main_hydra()