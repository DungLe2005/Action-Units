import csv
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import torch

from datasets.disfa import DISFA


class _IdentityTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, value):
        return value


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


transforms_stub = types.ModuleType("torchvision.transforms")
transforms_stub.Compose = _Compose
transforms_stub.Resize = _IdentityTransform
transforms_stub.RandomHorizontalFlip = _IdentityTransform
transforms_stub.RandomRotation = _IdentityTransform
transforms_stub.ColorJitter = _IdentityTransform
transforms_stub.ToTensor = _IdentityTransform
transforms_stub.Normalize = _IdentityTransform
transforms_stub.Pad = _IdentityTransform
transforms_stub.RandomCrop = _IdentityTransform
torchvision_stub = types.ModuleType("torchvision")
torchvision_stub.transforms = transforms_stub
sys.modules.setdefault("torchvision", torchvision_stub)
sys.modules.setdefault("torchvision.transforms", transforms_stub)

timm_stub = types.ModuleType("timm")
timm_data_stub = types.ModuleType("timm.data")
timm_random_erasing_stub = types.ModuleType("timm.data.random_erasing")
timm_random_erasing_stub.RandomErasing = _IdentityTransform
sys.modules.setdefault("timm", timm_stub)
sys.modules.setdefault("timm.data", timm_data_stub)
sys.modules.setdefault("timm.data.random_erasing", timm_random_erasing_stub)

from datasets.make_dataloader import (
    build_disfa_subject_folds,
    compute_au_pos_weight,
    make_au_dataloader,
)
from loss.make_loss import make_loss
from utils.au_fold_report import flatten_fold_metrics, summarize_fold_rows
from utils.au_training_history import history_row, write_stage2_history


AU_COLUMNS = [
    "AU1",
    "AU2",
    "AU4",
    "AU5",
    "AU6",
    "AU9",
    "AU12",
    "AU15",
    "AU17",
    "AU20",
    "AU25",
    "AU26",
]


def _write_labels_csv(root):
    os.makedirs(root, exist_ok=True)
    rows = []
    for subject_index in range(1, 7):
        subject = f"SN{subject_index:03d}"
        for frame_index in range(2):
            row = {
                "image_path": f"{subject}/Trial_1/{frame_index:03d}.jpg",
            }
            for au_name in AU_COLUMNS:
                row[au_name] = 0
            rows.append(row)

    rows[0]["AU1"] = 1
    rows[2]["AU4"] = 1
    rows[4]["AU6"] = 1

    with open(os.path.join(root, "labels.csv"), "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path"] + AU_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _cfg(root):
    return SimpleNamespace(
        DATASETS=SimpleNamespace(ROOT_DIR=root),
        DATALOADER=SimpleNamespace(NUM_WORKERS=0),
        SOLVER=SimpleNamespace(IMS_PER_BATCH=2, SEED=42),
        TEST=SimpleNamespace(IMS_PER_BATCH=2),
        INPUT=SimpleNamespace(SIZE_TRAIN=[224, 224]),
    )


class TestDISFAProtocol(unittest.TestCase):
    def test_subject_folds_are_deterministic_and_exclusive(self):
        with tempfile.TemporaryDirectory() as root:
            _write_labels_csv(root)
            folds_1 = build_disfa_subject_folds(root, num_folds=3, seed=123)
            folds_2 = build_disfa_subject_folds(root, num_folds=3, seed=123)

            self.assertEqual(folds_1, folds_2)
            self.assertEqual(len(folds_1), 3)
            self.assertEqual(
                sorted(subject for fold in folds_1 for subject in fold),
                [f"SN{index:03d}" for index in range(1, 7)],
            )
            for index, fold in enumerate(folds_1):
                train_subjects = {
                    subject
                    for other_index, other_fold in enumerate(folds_1)
                    if other_index != index
                    for subject in other_fold
                }
                self.assertFalse(train_subjects.intersection(fold))

    def test_make_au_dataloader_uses_distinct_train_and_val_datasets(self):
        with tempfile.TemporaryDirectory() as root:
            _write_labels_csv(root)
            train_loader, val_loader, num_aus, pos_weight, fold_info = make_au_dataloader(
                _cfg(root), fold_idx=0
            )

            self.assertEqual(num_aus, 12)
            self.assertEqual(len(pos_weight), 12)
            self.assertIn("train_subjects", fold_info)
            self.assertIn("val_subjects", fold_info)
            self.assertIsNot(train_loader.dataset, val_loader.dataset)
            self.assertIsNot(train_loader.dataset.transform, val_loader.dataset.transform)
            self.assertFalse(
                set(fold_info["train_subjects"]).intersection(fold_info["val_subjects"])
            )

    def test_make_au_dataloader_prefers_stage2_batch_size(self):
        with tempfile.TemporaryDirectory() as root:
            _write_labels_csv(root)
            cfg = _cfg(root)
            cfg.SOLVER.STAGE2 = SimpleNamespace(IMS_PER_BATCH=3)

            train_loader, _, _, _, _ = make_au_dataloader(cfg, fold_idx=0)

            self.assertEqual(train_loader.batch_size, 3)

    def test_pos_weight_uses_train_split_labels(self):
        with tempfile.TemporaryDirectory() as root:
            _write_labels_csv(root)
            dataset = DISFA(root=root, subjects=["SN001", "SN002"])
            pos_weight = compute_au_pos_weight(dataset)

            self.assertAlmostEqual(float(pos_weight[0]), 3.0)
            self.assertAlmostEqual(float(pos_weight[1]), 1.0)

    def test_make_loss_uses_supplied_pos_weight(self):
        cfg = SimpleNamespace(
            DATASETS=SimpleNamespace(NAMES="disfa"),
            DATALOADER=SimpleNamespace(SAMPLER="softmax"),
            MODEL=SimpleNamespace(
                METRIC_LOSS_TYPE="triplet",
                NO_MARGIN=False,
                IF_LABELSMOOTH="off",
            ),
            SOLVER=SimpleNamespace(MARGIN=0.3),
        )
        pos_weight = torch.ones(12) * 2
        loss_func, _ = make_loss(
            cfg, num_classes=12, pos_weight=pos_weight, device="cpu"
        )

        self.assertEqual(loss_func.bce.pos_weight.tolist(), pos_weight.tolist())

    def test_fold_report_summary_keeps_disfa8_keys(self):
        metrics_1 = {
            "disfa8_f1_macro": 0.5,
            "disfa8_auc_macro": 0.7,
            "avg_f1": 0.4,
            "avg_auc": 0.6,
            "accuracy": 0.8,
            "f1_macro": 0.4,
            "f1_micro": 0.45,
            "disfa8_per_au_f1": {au_name: 0.5 for au_name in AU_COLUMNS},
        }
        metrics_2 = dict(metrics_1)
        metrics_2["disfa8_f1_macro"] = 0.7
        metrics_2["disfa8_per_au_f1"] = {au_name: 0.7 for au_name in AU_COLUMNS}

        fold_info_1 = {
            "fold_idx": 0,
            "train_subjects": ["SN001"],
            "val_subjects": ["SN002"],
            "train_samples": 2,
            "val_samples": 2,
        }
        fold_info_2 = dict(fold_info_1)
        fold_info_2["fold_idx"] = 1

        rows = [
            flatten_fold_metrics(metrics_1, fold_info_1),
            flatten_fold_metrics(metrics_2, fold_info_2),
        ]
        summary = summarize_fold_rows(rows)

        self.assertAlmostEqual(summary["mean"]["disfa8_f1_macro"], 0.6)
        self.assertIn("disfa8_f1_AU1", rows[0])
        self.assertIn("disfa8_f1_AU26", rows[0])

    def test_stage2_history_writes_metric_files(self):
        metrics = {
            "disfa8_f1_macro": 0.6,
            "disfa8_auc_macro": 0.7,
            "avg_f1": 0.5,
            "avg_auc": 0.65,
            "accuracy": 0.8,
            "f1_macro": 0.45,
            "f1_micro": 0.55,
        }

        with tempfile.TemporaryDirectory() as root:
            row = history_row(
                epoch=1,
                train_loss=2.0,
                lr=3.5e-6,
                metrics=metrics,
                best_metric=0.6,
                is_best=True,
                itc_enabled=False,
                grad_norm=1.25,
                max_logit_abs=3.5,
                train_positive_rate=0.42,
            )
            paths = write_stage2_history(root, [row])

            self.assertTrue(os.path.exists(paths["csv_path"]))
            self.assertTrue(os.path.exists(paths["json_path"]))
            with open(paths["csv_path"], newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertEqual(rows[0]["epoch"], "1")
            self.assertEqual(rows[0]["is_best"], "True")
            self.assertEqual(rows[0]["grad_norm"], "1.25")
            self.assertEqual(rows[0]["max_logit_abs"], "3.5")
            self.assertEqual(rows[0]["stopped_early"], "False")


if __name__ == "__main__":
    unittest.main()
