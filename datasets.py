from packages.MultiBench.datasets.avmnist.get_data import get_dataloader as avmnist_data_loader
from packages.MultiBench.datasets.imdb.get_data import get_dataloader as mmimdb_data_loader

DATASET_DICT = {
    "avmnist": avmnist_data_loader,
    "mmimdb": mmimdb_data_loader
}

def get_dataloader(cfg):
    dname, data_dir = cfg.experiment.dataset.name, cfg.experiment.dataset.path
    if dname not in DATASET_DICT:
        raise NotImplementedError(f"Dataset {dname} not yet supported.")
    return DATASET_DICT[dname](data_dir, batch_size=cfg.batch_size, **cfg.experiment.dataset.args)
    