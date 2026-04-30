#!/usr/bin/env python3
"""Train and select a Poker44 miner model from the public benchmark artifact."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
from poker44_ml.features import chunk_features
from poker44_ml.inference import Poker44Model
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional speed/quality boost.
    XGBClassifier = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a calibrated chunk-level Poker44 miner model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/public_miner_benchmark.json.gz"),
        help="Path to the public miner benchmark JSON or JSON.gz file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/poker44_xgb_calibrated.joblib"),
        help="Path where the trained joblib artifact should be saved.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", "--random-state", dest="seed", type=int, default=44)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument(
        "--calibration",
        choices=("auto", "isotonic", "sigmoid", "none"),
        default="auto",
    )
    parser.add_argument(
        "--selection-objective",
        choices=("balanced", "low_fpr"),
        default="low_fpr",
    )
    parser.add_argument(
        "--recall-target",
        type=float,
        default=0.9,
        help="Recall target used when measuring false-positive rate.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Evaluate multiple model configurations and save the best one.",
    )
    parser.add_argument(
        "--search-budget",
        type=int,
        default=6,
        help="Maximum candidate configurations to evaluate when --search is enabled.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Build it first with "
            "python scripts/publish/publish_public_benchmark.py --skip-wandb "
            "--output-path data/public_miner_benchmark.json.gz"
        )

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def benchmark_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    chunks = list(payload.get("labeled_chunks") or [])
    if not chunks:
        raise ValueError("Dataset contains no labeled_chunks.")

    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        hands = chunk.get("hands") or []
        if not hands:
            continue
        rows.append(
            {
                "features": chunk_features(hands),
                "label": int(bool(chunk.get("is_bot"))),
                "split": chunk.get("split") or "",
                "hands": hands,
            }
        )
    if not rows:
        raise ValueError("Dataset has no non-empty chunks.")

    feature_names = sorted(
        {name for row in rows for name in row["features"].keys()}
    )
    if not feature_names:
        raise ValueError("Feature extraction produced no features.")
    return rows, feature_names


def split_indices(
    rows: list[dict[str, Any]],
    *,
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    train_idx = [idx for idx, row in enumerate(rows) if row["split"] == "train"]
    validation_idx = [
        idx for idx, row in enumerate(rows) if row["split"] == "validation"
    ]
    if train_idx and validation_idx:
        return train_idx, validation_idx

    labels = [row["label"] for row in rows]
    indices = list(range(len(rows)))
    stratify = labels if len(set(labels)) > 1 and min(Counter(labels).values()) >= 2 else None
    train_idx, validation_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return list(train_idx), list(validation_idx)


def matrix(
    rows: list[dict[str, Any]],
    indices: list[int],
    feature_names: list[str],
) -> tuple[list[list[float]], list[int]]:
    x = [
        [float(rows[idx]["features"].get(name, 0.0)) for name in feature_names]
        for idx in indices
    ]
    y = [int(rows[idx]["label"]) for idx in indices]
    return x, y


def false_positive_rate_at_threshold(
    y_true: list[int],
    y_prob: list[float],
    *,
    threshold: float = 0.5,
) -> float:
    negative_count = sum(1 for label in y_true if label == 0)
    if negative_count == 0:
        return 0.0
    false_positives = sum(
        1 for label, prob in zip(y_true, y_prob) if label == 0 and prob >= threshold
    )
    return false_positives / negative_count


def false_positive_rate_at_recall(
    y_true: list[int],
    y_prob: list[float],
    *,
    target_recall: float,
) -> dict[str, float]:
    if len(set(y_true)) < 2:
        return {
            "threshold": 0.5,
            "recall": 0.0,
            "fpr": false_positive_rate_at_threshold(y_true, y_prob),
        }

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "recall": 0.0,
            "fpr": false_positive_rate_at_threshold(y_true, y_prob),
        }

    recall_values = list(recall[:-1])
    best_index = min(
        range(len(thresholds)),
        key=lambda index: abs(float(recall_values[index]) - target_recall),
    )
    threshold = float(thresholds[best_index])
    predicted_positive = [prob >= threshold for prob in y_prob]
    positive_count = max(sum(1 for label in y_true if label == 1), 1)
    negative_count = max(sum(1 for label in y_true if label == 0), 1)
    true_positives = sum(
        1 for label, pred in zip(y_true, predicted_positive) if label == 1 and pred
    )
    false_positives = sum(
        1 for label, pred in zip(y_true, predicted_positive) if label == 0 and pred
    )
    return {
        "threshold": threshold,
        "recall": true_positives / positive_count,
        "fpr": false_positives / negative_count,
    }


def evaluate_predictions(
    y_true: list[int],
    y_prob: list[float],
    *,
    recall_target: float,
    latency_per_chunk_ms: float | None = None,
) -> dict[str, float]:
    clipped = [min(max(float(prob), 1e-6), 1.0 - 1e-6) for prob in y_prob]
    predictions = [1 if prob >= 0.5 else 0 for prob in clipped]
    recall_stats = false_positive_rate_at_recall(
        y_true,
        clipped,
        target_recall=recall_target,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
        "fpr_at_recall": float(recall_stats["fpr"]),
        "recall_target": float(recall_target),
        "threshold_at_recall": float(recall_stats["threshold"]),
        "achieved_recall": float(recall_stats["recall"]),
        "fpr_at_threshold_0_5": float(
            false_positive_rate_at_threshold(y_true, clipped, threshold=0.5)
        ),
    }
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, clipped))
        metrics["pr_auc"] = float(average_precision_score(y_true, clipped))
    else:
        metrics["roc_auc"] = 0.5
        metrics["pr_auc"] = float(sum(y_true) / max(len(y_true), 1))
    if latency_per_chunk_ms is not None and not math.isnan(latency_per_chunk_ms):
        metrics["latency_per_chunk_ms"] = float(latency_per_chunk_ms)
    return metrics


def format_metrics(metrics: dict[str, Any]) -> str:
    ordered = (
        "accuracy",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
        "fpr_at_recall",
        "achieved_recall",
        "threshold_at_recall",
        "fpr_at_threshold_0_5",
        "latency_per_chunk_ms",
    )
    return " | ".join(
        f"{key}={float(metrics[key]):.6f}" for key in ordered if key in metrics
    )


def model_selection_score(metrics: dict[str, float], objective: str) -> float:
    if objective == "balanced":
        return (
            0.45 * metrics["roc_auc"]
            + 0.30 * metrics["pr_auc"]
            - 0.20 * metrics["log_loss"]
            - 0.15 * metrics["brier_score"]
            - 0.30 * metrics["fpr_at_threshold_0_5"]
            - 0.35 * metrics["fpr_at_recall"]
        )
    return (
        0.35 * metrics["roc_auc"]
        + 0.20 * metrics["pr_auc"]
        - 0.15 * metrics["log_loss"]
        - 0.10 * metrics["brier_score"]
        - 0.60 * metrics["fpr_at_threshold_0_5"]
        - 0.80 * metrics["fpr_at_recall"]
    )


def choose_calibration(method: str, train_labels: list[int]) -> list[str | None]:
    if method == "none":
        return [None]
    counts = Counter(train_labels)
    min_class_count = min(counts.values()) if counts else 0
    available = ["sigmoid"]
    if min_class_count >= 12:
        available.append("isotonic")
    if method == "auto":
        return available + [None]
    if method == "isotonic" and min_class_count < 12:
        print("warning=isotonic_calibration_requires_more_rows falling_back=sigmoid")
        return ["sigmoid"]
    return [method]


def build_base_model(config: dict[str, float | int], seed: int) -> tuple[object, str]:
    if XGBClassifier is not None:
        booster = XGBClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            learning_rate=float(config["learning_rate"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
        forest = ExtraTreesClassifier(
            n_estimators=max(200, int(config["n_estimators"])),
            max_depth=int(config["max_depth"]) + 2,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
        return (
            VotingClassifier(
                estimators=[("xgb", booster), ("et", forest)],
                voting="soft",
                weights=[2, 1],
            ),
            "xgboost+extra-trees+sklearn-calibration",
        )

    booster = HistGradientBoostingClassifier(
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        max_iter=int(config["n_estimators"]),
        random_state=seed,
    )
    forest = ExtraTreesClassifier(
        n_estimators=max(300, int(config["n_estimators"])),
        max_depth=int(config["max_depth"]) + 2,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    return (
        VotingClassifier(
            estimators=[("hgb", booster), ("et", forest)],
            voting="soft",
            weights=[2, 1],
        ),
        "sklearn-hist-gradient-boosting+extra-trees+calibration",
    )


def build_search_configs(args: argparse.Namespace) -> list[dict[str, float | int]]:
    base = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    candidates = [
        base,
        {
            **base,
            "learning_rate": 0.03,
            "n_estimators": max(args.n_estimators, 450),
            "max_depth": max(args.max_depth, 4),
        },
        {
            **base,
            "learning_rate": 0.04,
            "n_estimators": max(args.n_estimators, 500),
            "max_depth": args.max_depth + 1,
        },
        {
            **base,
            "learning_rate": 0.025,
            "n_estimators": max(args.n_estimators, 650),
            "max_depth": max(args.max_depth, 6),
            "subsample": min(1.0, max(args.subsample, 0.95)),
            "colsample_bytree": min(1.0, max(args.colsample_bytree, 0.95)),
        },
        {
            **base,
            "learning_rate": 0.035,
            "n_estimators": max(args.n_estimators, 700),
            "max_depth": max(args.max_depth, 6),
        },
        {
            **base,
            "learning_rate": 0.02,
            "n_estimators": max(args.n_estimators, 800),
            "max_depth": max(args.max_depth, 7),
        },
    ]
    return candidates[: max(1, int(args.search_budget))]


def candidate_models(
    *,
    config: dict[str, float | int],
    train_labels: list[int],
    calibration: str,
    seed: int,
) -> list[tuple[object, str, str]]:
    models = []
    for calibration_method in choose_calibration(calibration, train_labels):
        base_model, framework_name = build_base_model(config, seed)
        if calibration_method is None:
            models.append((base_model, "none", framework_name))
            continue
        min_class_count = min(Counter(train_labels).values())
        cv = min(3, min_class_count)
        if cv < 2:
            models.append((base_model, "none", framework_name))
            continue
        models.append(
            (
                CalibratedClassifierCV(base_model, method=calibration_method, cv=cv),
                calibration_method,
                framework_name,
            )
        )
    return models


def fit_best_model(
    *,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    feature_names: list[str],
) -> tuple[object, dict[str, float], dict[str, Any], list[int], list[int]]:
    train_idx, validation_idx = split_indices(rows, test_size=args.test_size, seed=args.seed)
    x_train, y_train = matrix(rows, train_idx, feature_names)
    x_validation, y_validation = matrix(rows, validation_idx, feature_names)

    configs = build_search_configs(args) if args.search else [
        {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
        }
    ]

    best_model = None
    best_metrics: dict[str, float] = {}
    best_config: dict[str, Any] = {}
    best_score = float("-inf")

    for config_index, config in enumerate(configs, start=1):
        print(f"search_candidate={config_index}/{len(configs)} config={config}")
        for model, calibration_method, framework_name in candidate_models(
            config=config,
            train_labels=y_train,
            calibration=args.calibration,
            seed=args.seed,
        ):
            started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - started
            probabilities = model.predict_proba(x_validation)[:, 1]
            metrics = evaluate_predictions(
                y_validation,
                list(probabilities),
                recall_target=args.recall_target,
            )
            score = model_selection_score(metrics, args.selection_objective)
            print(
                "candidate",
                f"calibration={calibration_method}",
                f"fit_seconds={fit_seconds:.3f}",
                f"selection_score={score:.6f}",
                format_metrics(metrics),
            )
            if score > best_score:
                best_score = score
                best_model = model
                best_metrics = dict(metrics)
                best_config = {
                    **config,
                    "calibration": calibration_method,
                    "framework": framework_name,
                    "fit_seconds": fit_seconds,
                    "selection_score": score,
                }

    if best_model is None:
        raise RuntimeError("No model candidate was trained.")
    return best_model, best_metrics, best_config, train_idx, validation_idx


def add_latency_metric(
    output_path: Path,
    rows: list[dict[str, Any]],
    validation_idx: list[int],
    metrics: dict[str, float],
) -> dict[str, float]:
    sample_indices = validation_idx[: min(8, len(validation_idx))]
    if not sample_indices:
        return metrics
    loaded = Poker44Model(output_path)
    latency = loaded.benchmark_latency(
        [rows[idx]["hands"] for idx in sample_indices],
        repeats=3,
    )
    enriched = dict(metrics)
    enriched["latency_per_chunk_ms"] = latency["latency_per_chunk_ms"]
    return enriched


def main() -> None:
    args = parse_args()
    payload = load_payload(args.dataset)
    rows, feature_names = benchmark_rows(payload)
    model, metrics, config, train_idx, validation_idx = fit_best_model(
        args=args,
        rows=rows,
        feature_names=feature_names,
    )

    metadata = {
        "dataset_hash": payload.get("dataset_hash"),
        "dataset_path": str(args.dataset),
        "model_name": "poker44-public-ensemble",
        "model_version": "local-2",
        "framework": config.get("framework", "sklearn"),
        "train_chunks": float(len(train_idx)),
        "validation_chunks": float(len(validation_idx)),
        "feature_count": float(len(feature_names)),
        "selection_objective": args.selection_objective,
        "recall_target": float(args.recall_target),
        **{key: value for key, value in config.items() if isinstance(value, (int, float, str))},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metadata": metadata,
        },
        args.output,
    )
    metrics = add_latency_metric(args.output, rows, validation_idx, metrics)

    print(f"saved={args.output}")
    print(f"feature_count={len(feature_names)}")
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
