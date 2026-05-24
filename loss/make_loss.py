# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .au_loss import WeightedBCELoss
import torch


def _cfg_value(section, name, default):
    return getattr(section, name, default)


def _adjust_disfa_pos_weight(cfg, pos_weight):
    if pos_weight is None:
        return None

    stage2_cfg = getattr(cfg.SOLVER, "STAGE2", None)
    power = float(_cfg_value(stage2_cfg, "POS_WEIGHT_POWER", 1.0))
    max_value = float(_cfg_value(stage2_cfg, "POS_WEIGHT_MAX", 0.0))

    adjusted = pos_weight.float()
    if power != 1.0:
        adjusted = adjusted.pow(power)
    if max_value > 0.0:
        adjusted = torch.clamp(adjusted, max=max_value)
    return adjusted


def make_loss(cfg, num_classes, pos_weight=None, device=None):  # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    use_gpu = torch.cuda.is_available() if device is None else device == "cuda"
    center_criterion = CenterLoss(
        num_classes=num_classes, feat_dim=feat_dim, use_gpu=use_gpu
    )  # center loss
    if "triplet" in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print(
            "expected METRIC_LOSS_TYPE should be triplet"
            "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE)
        )

    if cfg.MODEL.IF_LABELSMOOTH == "on":
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if cfg.DATASETS.NAMES == "disfa":
        if pos_weight is not None:
            if device is not None:
                pos_weight = pos_weight.to(device)
            raw_pos_weight = pos_weight
            pos_weight = _adjust_disfa_pos_weight(cfg, pos_weight)
            print(f"Using train-split pos_weight for DISFA: {raw_pos_weight.tolist()}")
            if not torch.equal(pos_weight, raw_pos_weight):
                print(
                    "Using adjusted Stage 2 pos_weight for DISFA: "
                    f"{pos_weight.tolist()}"
                )
        else:
            print("Using unweighted BCE for DISFA; no train-split pos_weight was provided")
        loss_func = WeightedBCELoss(pos_weight=pos_weight)
    elif sampler == "softmax":

        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == "softmax_triplet":

        def loss_func(score, feat, target, target_cam, i2tscore=None):
            if cfg.MODEL.METRIC_LOSS_TYPE == "triplet":
                if cfg.MODEL.IF_LABELSMOOTH == "on":
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = (
                        cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                        + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    )

                    if i2tscore != None:
                        I2TLOSS = xent(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    return loss
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                        ID_LOSS = sum(ID_LOSS)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                        TRI_LOSS = [triplet(feats, target)[0] for feats in feat[0:]]
                        TRI_LOSS = sum(TRI_LOSS)
                    else:
                        TRI_LOSS = triplet(feat, target)[0]

                    loss = (
                        cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS
                        + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                    )

                    if i2tscore != None:
                        I2TLOSS = F.cross_entropy(i2tscore, target)
                        loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                    return loss
            else:
                print(
                    "expected METRIC_LOSS_TYPE should be triplet"
                    "but got {}".format(cfg.MODEL.METRIC_LOSS_TYPE)
                )

    else:
        print(
            "expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center"
            "but got {}".format(cfg.DATALOADER.SAMPLER)
        )
    return loss_func, center_criterion
