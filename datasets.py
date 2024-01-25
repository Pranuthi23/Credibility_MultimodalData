from packages.MultiBench.datasets.avmnist.get_data import get_dataloader as avmnist_data_loader
from packages.MultiBench.datasets.imdb.get_data import get_dataloader as mmimdb_data_loader
from data.cub.get_data import get_dataloader as cub_data_loader
from data.sunrgbd.get_data import get_dataloader as sunrgbd_data_loader
from data.nyud2.get_data import get_dataloader as nyud2_data_loader
from data.sunrgb_d.get_data import get_dataloader as sunrgb_d_data_loader
from data.multiview3D.get_data import get_dataloader as multiview_data_loader


DATASET_DICT = {
    "avmnist": avmnist_data_loader,
    "mmimdb": mmimdb_data_loader,
    "cub": cub_data_loader,
    "nyud2": nyud2_data_loader,
    "sunrgb_d": sunrgb_d_data_loader,
    "multiview": multiview_data_loader
}

def get_dataloader(cfg):
    dname, data_dir = cfg.experiment.dataset.name, cfg.experiment.dataset.path
    if dname not in DATASET_DICT:
        raise NotImplementedError(f"Dataset {dname} not yet supported.")
    return DATASET_DICT[dname](data_dir, batch_size=cfg.batch_size, **cfg.experiment.dataset.args)
    