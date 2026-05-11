import csv
import json
import math
import os


DISFA8_KEYS = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
METRIC_KEYS = [
    "disfa8_f1_macro",
    "disfa8_auc_macro",
    "avg_f1",
    "avg_auc",
    "accuracy",
    "f1_macro",
    "f1_micro",
]


def flatten_fold_metrics(metrics, fold_info):
    row = {
        "fold": fold_info["fold_idx"],
        "train_subjects": "|".join(fold_info["train_subjects"]),
        "val_subjects": "|".join(fold_info["val_subjects"]),
        "train_samples": fold_info["train_samples"],
        "val_samples": fold_info["val_samples"],
    }

    for key in METRIC_KEYS:
        row[key] = float(metrics.get(key, float("nan")))

    disfa8_f1 = metrics.get("disfa8_per_au_f1", {})
    for au_name in DISFA8_KEYS:
        row[f"disfa8_f1_{au_name}"] = float(disfa8_f1.get(au_name, float("nan")))

    return row


def _mean(values):
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


def _std(values):
    valid = [value for value in values if not math.isnan(value)]
    if len(valid) < 2:
        return float("nan")
    mean_value = _mean(valid)
    variance = sum((value - mean_value) ** 2 for value in valid) / (len(valid) - 1)
    return math.sqrt(variance)


def summarize_fold_rows(rows):
    metric_columns = [
        key
        for key in rows[0].keys()
        if key not in {"fold", "train_subjects", "val_subjects"}
    ]
    summary = {"mean": {"fold": "mean"}, "std": {"fold": "std"}}

    for subject_key in ["train_subjects", "val_subjects"]:
        summary["mean"][subject_key] = ""
        summary["std"][subject_key] = ""

    for column in metric_columns:
        values = [float(row[column]) for row in rows]
        summary["mean"][column] = _mean(values)
        summary["std"][column] = _std(values)

    return summary


def write_fold_reports(output_dir, fold_records):
    rows = [record["row"] for record in fold_records]
    summary = summarize_fold_rows(rows)
    csv_rows = rows + [summary["mean"], summary["std"]]

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "fold_metrics.csv")
    json_path = os.path.join(output_dir, "fold_metrics.json")

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        "primary_metric": "disfa8_f1_macro",
        "folds": fold_records,
        "summary": summary,
    }
    with open(json_path, "w") as json_file:
        json.dump(report, json_file, indent=2, allow_nan=True)

    return {"csv_path": csv_path, "json_path": json_path, "summary": summary}
