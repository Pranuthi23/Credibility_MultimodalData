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

# A logger for this file
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

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
    model_class = LateFusionMultiLabelClassifier if(cfg.experiment.multilabel) else LateFusionClassifier
    if cfg.load_and_eval:
        model = model_class.load_from_checkpoint(f"{run_dir}/best_model.pt")
    else:
        if cfg.experiment.classification:
            if(cfg.experiment.multilabel):
                model = LateFusionMultiLabelClassifier(cfg, steps_per_epoch=len(train_loader))
            else:     
                model = LateFusionClassifier(cfg, steps_per_epoch=len(train_loader))
            
        if cfg.torch_compile:  
            # model = torch.compile(model)
            raise NotImplementedError("Torch compilation not yet supported.")
        
    # Store number of model parameters
    summary = ModelSummary(model, max_depth=-1)
    logger.info("Model:")
    logger.info(model)
    logger.info("Summary:")
    logger.info(summary)

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

    if not cfg.load_and_eval:
        # Fit model
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) 

    logger.info("Evaluating model...")
    trainer.test(model=model, dataloaders=[train_loader, val_loader, test_loader], verbose=True)
    logger.info("Finished evaluation...")

    # Save checkpoint in general models directory to be used across experiments
    chpt_path = os.path.join(run_dir, "best_model.pt")
    logger.info("Saving checkpoint: " + chpt_path)
    trainer.save_checkpoint(chpt_path)


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
        cfg.tag = cfg.experiment.name

    if "group_tag" not in cfg:
        cfg.group_tag = cfg.dataset

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