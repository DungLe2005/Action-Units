import csv
import json
import math
import numbers
import os


HISTORY_METRIC_KEYS = [
    "train_loss",
    "lr",
    "disfa8_f1_macro",
    "disfa8_auc_macro",
    "avg_f1",
    "avg_auc",
    "accuracy",
    "f1_macro",
    "f1_micro",
]


def _clean_float(value):
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def history_row(epoch, train_loss, lr, metrics, best_metric, is_best, itc_enabled):
    row = {
        "epoch": int(epoch),
        "train_loss": _clean_float(train_loss),
        "lr": _clean_float(lr),
        "best_metric": _clean_float(best_metric),
        "is_best": bool(is_best),
        "itc_enabled": bool(itc_enabled),
    }
    for key in HISTORY_METRIC_KEYS:
        if key in {"train_loss", "lr"}:
            continue
        row[key] = _clean_float(metrics.get(key))
    for key, value in metrics.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            row.setdefault(key, _clean_float(value))
        elif isinstance(value, dict):
            for item_key, item_value in value.items():
                if isinstance(item_value, numbers.Real) and not isinstance(
                    item_value, bool
                ):
                    row.setdefault(
                        "{}_{}".format(key, item_key), _clean_float(item_value)
                    )
    return row


def write_stage2_history(output_dir, records, prefix="stage2"):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{prefix}_history.csv")
    json_path = os.path.join(output_dir, f"{prefix}_history.json")
    plot_path = os.path.join(output_dir, f"{prefix}_metric_curves.png")

    if records:
        fieldnames = []
        for record in records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(csv_path, "w", newline="") as csv_file:
            csv_file.write("")

    with open(json_path, "w") as json_file:
        json.dump(records, json_file, indent=2, allow_nan=True)

    plotted = _plot_stage2_history(records, plot_path)
    return {
        "csv_path": csv_path,
        "json_path": json_path,
        "plot_path": plot_path if plotted else None,
    }


def _plot_stage2_history(records, plot_path):
    if not records:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    epochs = [record["epoch"] for record in records]
    metric_groups = [
        ("Loss", ["train_loss"]),
        ("F1", ["disfa8_f1_macro", "avg_f1", "f1_macro", "f1_micro"]),
        ("AUC / Accuracy", ["disfa8_auc_macro", "avg_auc", "accuracy"]),
        ("LR", ["lr"]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.reshape(-1)

    for axis, (title, keys) in zip(axes, metric_groups):
        for key in keys:
            values = [_clean_float(record.get(key)) for record in records]
            if all(math.isnan(value) for value in values):
                continue
            axis.plot(epochs, values, marker="o", linewidth=1.5, label=key)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        if axis.lines:
            axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True
