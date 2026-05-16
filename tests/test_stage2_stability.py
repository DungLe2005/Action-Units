from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from loss.au_loss import WeightedBCELoss
from processor.processor_au_2stage import (
    _evaluate_stage1_itc_loss,
    _update_stage2_early_stop,
)
from solver.lr_scheduler import WarmupMultiStepLR
from solver.make_optimizer_prompt import make_optimizer_2stage


class _DummyStage2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        self.bottleneck = nn.BatchNorm1d(4)
        self.classifier = nn.Linear(4, 12)
        self.prompt_learner = nn.Linear(4, 4)
        self.text_encoder = nn.Linear(4, 4)


class _DummyStage1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_features = nn.Parameter(torch.eye(12))

    def forward(self, x=None, get_image=False, get_text=False, **kwargs):
        if get_text:
            return self.text_features
        if get_image:
            return x
        raise AssertionError("Unexpected Stage 1 dummy call")


def _stage2_cfg():
    return SimpleNamespace(
        SOLVER=SimpleNamespace(
            STAGE2=SimpleNamespace(
                BASE_LR=5e-5,
                BACKBONE_LR_FACTOR=0.1,
                OPTIMIZER_NAME="Adam",
                WEIGHT_DECAY=1e-4,
                CENTER_LR=0.5,
                BIAS_LR_FACTOR=1,
                WEIGHT_DECAY_BIAS=5e-4,
                LARGE_FC_LR=False,
            )
        )
    )


class TestStage2Stability(unittest.TestCase):
    def test_dual_head_bce_is_averaged(self):
        loss = WeightedBCELoss()
        logits_a = torch.tensor([[0.0, 1.0], [-1.0, 2.0]])
        logits_b = torch.tensor([[2.0, -1.0], [0.5, -0.5]])
        targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        actual = loss([logits_a, logits_b], targets)
        expected = 0.5 * (loss.bce(logits_a, targets) + loss.bce(logits_b, targets))

        self.assertTrue(torch.allclose(actual, expected))

    def test_stage2_optimizer_uses_lower_backbone_lr(self):
        model = _DummyStage2Model()
        optimizer, _ = make_optimizer_2stage(_stage2_cfg(), model, nn.Linear(1, 1))

        groups = {group["name"]: group for group in optimizer.param_groups}

        self.assertAlmostEqual(groups["image_encoder.0.weight"]["lr"], 5e-6)
        self.assertEqual(groups["image_encoder.0.weight"]["stage2_group"], "backbone")
        self.assertAlmostEqual(groups["classifier.weight"]["lr"], 5e-5)
        self.assertEqual(groups["classifier.weight"]["stage2_group"], "head")
        self.assertFalse(model.prompt_learner.weight.requires_grad)
        self.assertFalse(model.text_encoder.weight.requires_grad)

    def test_stage2_warmup_lr_sequence(self):
        parameter = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([{"params": [parameter], "lr": 5e-5}])
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=[25, 40],
            gamma=0.1,
            warmup_factor=0.01,
            warmup_iters=5,
            warmup_method="linear",
        )

        lrs = [optimizer.param_groups[0]["lr"]]
        for _ in range(5):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        expected = [5e-7, 1.04e-5, 2.03e-5, 3.02e-5, 4.01e-5, 5e-5]
        for actual, expected_lr in zip(lrs, expected):
            self.assertAlmostEqual(actual, expected_lr)

    def test_stage2_early_stop_requires_min_delta(self):
        best, wait, improved = _update_stage2_early_stop(0.3000, -1.0, 0, 0.001)
        self.assertTrue(improved)
        self.assertAlmostEqual(best, 0.3000)
        self.assertEqual(wait, 0)

        best, wait, improved = _update_stage2_early_stop(0.3005, best, wait, 0.001)
        self.assertFalse(improved)
        self.assertAlmostEqual(best, 0.3000)
        self.assertEqual(wait, 1)

        best, wait, improved = _update_stage2_early_stop(0.3020, best, wait, 0.001)
        self.assertTrue(improved)
        self.assertAlmostEqual(best, 0.3020)
        self.assertEqual(wait, 0)

    def test_stage1_itc_eval_returns_finite_loss_and_restores_train_mode(self):
        model = _DummyStage1Model()
        model.train()
        images = torch.eye(12)[:2]
        targets = torch.eye(12)[:2]
        val_loader = [(images, targets, None, None, None)]

        loss = _evaluate_stage1_itc_loss(
            model=model,
            text_model=model,
            val_loader=val_loader,
            device="cpu",
            loss_fn_itc=nn.BCEWithLogitsLoss(),
            temperature=0.07,
            desc="Test Stage1 Eval",
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
