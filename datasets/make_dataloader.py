import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .disfa import DISFA
from .preprocessing import build_au_train_transforms, build_au_val_transforms

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'disfa': DISFA
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    
    train_set = ImageDataset(dataset.train, train_transforms)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num

def build_disfa_subject_folds(root, num_folds=3, seed=42):
    dataset = DISFA(root=root, transform=None)
    subjects = np.array(dataset.all_subjects, dtype=object)
    if len(subjects) < num_folds:
        raise ValueError(
            f"DISFA subject-exclusive split needs at least {num_folds} subjects, "
            f"got {len(subjects)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    return [sorted(fold.tolist()) for fold in np.array_split(subjects, num_folds)]


def compute_au_pos_weight(dataset):
    labels = dataset.df[dataset.au_cols].to_numpy(dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = np.divide(
        negatives,
        positives,
        out=np.ones_like(positives, dtype=np.float32),
        where=positives > 0,
    )
    zero_positive_aus = [
        dataset.au_cols[index]
        for index, positive_count in enumerate(positives)
        if positive_count == 0
    ]
    if zero_positive_aus:
        print(
            "Warning: no positive samples in train split for "
            f"{zero_positive_aus}; using pos_weight=1.0 for those AUs"
        )
    return torch.tensor(weights, dtype=torch.float32)


def make_au_dataloader(cfg, fold_idx=0, num_folds=3):
    train_transforms = build_au_train_transforms(cfg)
    val_transforms = build_au_val_transforms(cfg)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    train_batch_size = cfg.SOLVER.IMS_PER_BATCH
    if hasattr(cfg.SOLVER, "STAGE2"):
        train_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH

    folds = build_disfa_subject_folds(
        root=cfg.DATASETS.ROOT_DIR,
        num_folds=num_folds,
        seed=cfg.SOLVER.SEED,
    )
    fold_idx = int(fold_idx)
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"fold_idx must be in [0, {len(folds) - 1}], got {fold_idx}")

    val_subjects = folds[fold_idx]
    train_subjects = [
        subject
        for index, fold in enumerate(folds)
        if index != fold_idx
        for subject in fold
    ]
    overlap = set(train_subjects).intersection(val_subjects)
    if overlap:
        raise AssertionError(f"DISFA train/val subject overlap: {sorted(overlap)}")

    train_set = DISFA(
        root=cfg.DATASETS.ROOT_DIR,
        transform=train_transforms,
        subjects=train_subjects,
    )
    val_set = DISFA(
        root=cfg.DATASETS.ROOT_DIR,
        transform=val_transforms,
        subjects=val_subjects,
    )
    pos_weight = compute_au_pos_weight(train_set)
    fold_info = {
        "fold_idx": fold_idx,
        "num_folds": num_folds,
        "train_subjects": sorted(train_subjects),
        "val_subjects": sorted(val_subjects),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "pos_weight": pos_weight.tolist(),
    }

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, 12, pos_weight, fold_info # 12 Action Units
