import math
import unittest

import numpy as np
import torch

from processor.processor_au import AUEvaluator
from processor.processor_au_2stage import _au_probabilities_from_logits


def _alternating_targets():
    pattern = (np.arange(12) % 2).astype(np.float32)
    return np.vstack([pattern, 1.0 - pattern, pattern, 1.0 - pattern])


class TestAUEvaluator(unittest.TestCase):
    def test_stage2_eval_converts_dual_head_logits_to_probabilities(self):
        logits_a = torch.zeros(2, 12)
        logits_b = torch.ones(2, 12) * 2.0

        probs = _au_probabilities_from_logits([logits_a, logits_b])

        self.assertEqual(tuple(probs.shape), (2, 12))
        self.assertTrue(torch.all(probs >= 0.0))
        self.assertTrue(torch.all(probs <= 1.0))
        self.assertTrue(torch.allclose(probs, torch.sigmoid(torch.ones(2, 12))))

    def test_perfect_prediction_scores_one(self):
        targets = _alternating_targets()
        probs = np.where(targets == 1.0, 0.9, 0.1)

        evaluator = AUEvaluator()
        evaluator.update(probs, targets)
        results = evaluator.compute()

        self.assertAlmostEqual(results["avg_f1"], 1.0)
        self.assertAlmostEqual(results["f1_macro"], 1.0)
        self.assertAlmostEqual(results["f1_micro"], 1.0)
        self.assertAlmostEqual(results["avg_auc"], 1.0)
        self.assertAlmostEqual(results["roc_auc_macro"], 1.0)
        self.assertAlmostEqual(results["accuracy"], 1.0)
        self.assertAlmostEqual(results["exact_match_accuracy"], 1.0)
        self.assertAlmostEqual(results["hamming_loss"], 0.0)
        self.assertAlmostEqual(results["disfa8_f1_macro"], 1.0)

    def test_all_zero_target_auc_is_nan_for_that_au(self):
        targets = _alternating_targets()
        targets[:, 0] = 0.0
        probs = np.where(targets == 1.0, 0.9, 0.1)

        evaluator = AUEvaluator()
        evaluator.update(torch.tensor(probs), torch.tensor(targets))
        results = evaluator.compute()

        self.assertTrue(math.isnan(results["per_au_auc"]["AU1"]))
        self.assertTrue(math.isnan(results["per_class_auc_AU_0"]))
        self.assertAlmostEqual(results["avg_auc"], 1.0)

    def test_cpu_torch_input_matches_numpy_input(self):
        targets = _alternating_targets()
        probs = np.where(targets == 1.0, 0.8, 0.2)

        numpy_evaluator = AUEvaluator()
        numpy_evaluator.update(probs, targets)
        numpy_results = numpy_evaluator.compute()

        torch_evaluator = AUEvaluator()
        torch_evaluator.update(torch.tensor(probs), torch.tensor(targets))
        torch_results = torch_evaluator.compute()

        self.assertEqual(numpy_results["per_au_f1"], torch_results["per_au_f1"])
        self.assertEqual(numpy_results["per_au_auc"], torch_results["per_au_auc"])
        self.assertAlmostEqual(numpy_results["accuracy"], torch_results["accuracy"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_gpu_input_matches_cpu_input(self):
        targets = _alternating_targets()
        probs = np.where(targets == 1.0, 0.75, 0.25)

        cpu_evaluator = AUEvaluator()
        cpu_evaluator.update(torch.tensor(probs), torch.tensor(targets))
        cpu_results = cpu_evaluator.compute()

        gpu_evaluator = AUEvaluator()
        gpu_evaluator.update(
            torch.tensor(probs, device="cuda"),
            torch.tensor(targets, device="cuda"),
        )
        gpu_results = gpu_evaluator.compute()

        self.assertEqual(cpu_results["per_au_f1"], gpu_results["per_au_f1"])
        self.assertEqual(cpu_results["per_au_auc"], gpu_results["per_au_auc"])
        self.assertAlmostEqual(cpu_results["accuracy"], gpu_results["accuracy"])

    def test_threshold_is_strictly_greater_than_point_five(self):
        targets = np.zeros((2, 12), dtype=np.float32)
        probs = np.full((2, 12), 0.1, dtype=np.float32)
        targets[:, 0] = [1.0, 0.0]
        probs[:, 0] = [0.5, 0.49]
        targets[:, 1] = [1.0, 0.0]
        probs[:, 1] = [0.5001, 0.49]

        evaluator = AUEvaluator()
        evaluator.update(probs, targets)
        results = evaluator.compute()

        self.assertAlmostEqual(results["per_au_f1"]["AU1"], 0.0)
        self.assertAlmostEqual(results["per_au_f1"]["AU2"], 1.0)

    def test_rejects_logits_and_intensity_targets(self):
        evaluator = AUEvaluator()
        targets = np.zeros((2, 12), dtype=np.float32)
        probs = np.zeros((2, 12), dtype=np.float32)

        with self.assertRaises(ValueError):
            evaluator.update(probs + 2.0, targets)

        with self.assertRaises(ValueError):
            evaluator.update(probs + 0.5, targets + 2.0)

    def test_integration_keys_for_stage2_logging(self):
        targets = _alternating_targets()
        probs = np.where(targets == 1.0, 0.9, 0.1)

        evaluator = AUEvaluator()
        evaluator.reset()
        evaluator.update(probs, targets)
        results = evaluator.compute()

        for key in ["avg_f1", "avg_auc", "accuracy"]:
            self.assertIn(key, results)
        for key in [
            "f1_macro",
            "f1_micro",
            "precision_macro",
            "recall_macro",
            "roc_auc_macro",
            "per_class_f1_AU_0",
            "per_class_precision_AU_0",
            "per_class_recall_AU_0",
        ]:
            self.assertIn(key, results)


if __name__ == "__main__":
    unittest.main()
