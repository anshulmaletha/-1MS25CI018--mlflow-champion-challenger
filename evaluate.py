"""
evaluate.py - Final Model Evaluation & MLflow Experiment Tracking Pipeline

Project: Champion–Challenger Model Comparison with Hugging Face and MLflow
Student USN: 1MS25CI018
Benchmark: SST-2 (Stanford Sentiment Treebank - Validation Split, 100 samples)
Seed: 42

Models:
  Model A (Initial Champion): distilbert-base-uncased-finetuned-sst-2-english
  Model B (Challenger):        siebert/sentiment-roberta-large-english

Task: Sentiment Classification (Binary: POSITIVE / NEGATIVE)

MLflow Experiment: hf-sentiment-champion-challenger
Runs:
  1. A_sentiment_distilbert (Champion)
  2. B_sentiment_roberta    (Challenger)
  3. compare_sentiment      (Comparison & Decision Summary)

Modes:
  --mode full : Full final benchmark evaluation (100 samples)
  --mode dev  : Fast local validation (e.g., initial development subset)
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

# Ensure UTF-8 encoding on Windows consoles so MLflow Unicode glyphs/emojis print cleanly
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# -----------------------------------------------------------------------------
# 1. Configuration & Constants
# -----------------------------------------------------------------------------
STUDENT_USN = "1MS25CI018"
TASK_NAME = "sentiment-classification"
DATASET_NAME = "data/eval_inputs.csv"
DATASET_SOURCE = "SST-2 (Stanford Sentiment Treebank - Validation Split)"
DATASET_VERSION = "stanfordnlp/sst2"
RANDOM_SEED = 42

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "hf-sentiment-champion-challenger"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_A_ROLE = "Champion"
MODEL_A_RUN_NAME = "A_sentiment_distilbert"

MODEL_B_ID = "siebert/sentiment-roberta-large-english"
MODEL_B_ROLE = "Challenger"
MODEL_B_RUN_NAME = "B_sentiment_roberta"

COMPARISON_RUN_NAME = "compare_sentiment"

# Predefined Champion-Challenger Decision Rule (Immutable)
PRIMARY_METRIC = "macro_f1"
IMPROVEMENT_MARGIN = 0.01  # Challenger must achieve at least +0.01 to win
METRIC_DIRECTION = "higher_is_better"
TIE_BREAKER = "lower_avg_latency_ms"

MAX_LENGTH = 512
VALID_LABELS = {"POSITIVE", "NEGATIVE"}


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds across libraries to guarantee deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# 2. MLflow Server Verification & Setup
# -----------------------------------------------------------------------------
def check_and_configure_mlflow(
    tracking_uri: str = MLFLOW_TRACKING_URI,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> tuple[str, MlflowClient]:
    """
    Verify the MLflow tracking server is reachable and configure the experiment.
    
    Fails immediately with a clear, actionable error message if the server
    is offline, preventing silent fallbacks to local directory tracking.
    """
    health_endpoint = f"{tracking_uri.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_endpoint, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Server responded with non-200 status code: {response.status}"
                )
    except Exception as err:
        raise ConnectionError(
            f"\n[ERROR] Unable to connect to MLflow tracking server at '{tracking_uri}'.\n"
            "An active MLflow tracking server is required.\n"
            "Start the tracking server using:\n"
            "  .\\.venv\\Scripts\\python -m mlflow server --host 127.0.0.1 --port 5000 "
            "--backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns\n"
            f"Underlying connection error: {err}"
        ) from err

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            tags={
                "student_usn": STUDENT_USN,
                "task": TASK_NAME,
            },
        )
        print(f"[INFO] Created MLflow experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        print(f"[INFO] Using MLflow experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name=experiment_name)
    return experiment_id, client


# -----------------------------------------------------------------------------
# 3. Dataset Validation
# -----------------------------------------------------------------------------
def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate the evaluation benchmark CSV.
    
    Validation Checks:
      1. File exists.
      2. Required columns present ('id', 'text', 'ground_truth').
      3. No null or empty text values.
      4. Ground-truth values strictly in {'POSITIVE', 'NEGATIVE'}.
      5. 'id' values are strictly unique.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = {"id", "text", "ground_truth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df["text"] = df["text"].astype(str).str.strip()
    if (df["text"] == "").any():
        raise ValueError("Dataset contains empty or whitespace-only text entries.")

    df["ground_truth"] = df["ground_truth"].astype(str).str.strip().str.upper()
    invalid_labels = set(df["ground_truth"]) - VALID_LABELS
    if invalid_labels:
        raise ValueError(
            f"Invalid ground-truth labels found: {invalid_labels}. Allowed: {VALID_LABELS}"
        )

    if not df["id"].is_unique:
        raise ValueError("Dataset 'id' column contains duplicate values.")

    return df


# -----------------------------------------------------------------------------
# 4. Output Label Normalization
# -----------------------------------------------------------------------------
def normalize_label(raw_label: str) -> str:
    """
    Normalize raw Hugging Face classification labels into standard POSITIVE/NEGATIVE.
    Handles variations like POSITIVE, NEGATIVE, LABEL_0, LABEL_1, POS, NEG.
    """
    cleaned = str(raw_label).strip().upper()
    if cleaned in {"POSITIVE", "LABEL_1", "POS", "1"}:
        return "POSITIVE"
    elif cleaned in {"NEGATIVE", "LABEL_0", "NEG", "0"}:
        return "NEGATIVE"
    raise ValueError(f"Encountered unexpected model label: '{raw_label}'")


# -----------------------------------------------------------------------------
# 5. Model Loading & Inference (Download-Protected)
# -----------------------------------------------------------------------------
def get_model_pipeline(model_id: str, offline_only: bool = True, device: int = -1):
    """
    Load model pipeline with offline-first caching protection.
    Prevents redundant multi-gigabyte re-downloads during development and evaluation.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, local_files_only=True
        )
        print(f"[INFO] Loaded '{model_id}' from local offline cache.")
    except Exception as e:
        if offline_only:
            raise RuntimeError(
                f"Model '{model_id}' is not in local cache and offline_only is enabled. "
                f"Error: {e}"
            ) from e
        print(f"[INFO] Downloading model '{model_id}' from Hugging Face Hub...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, local_files_only=False
        )

    return pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
    )


def run_model_inference(
    model_id: str,
    texts: list[str],
    offline_only: bool = True,
    device: int = -1,
) -> tuple[list[str], list[float], float, float]:
    """
    Execute real model forward passes on the evaluation texts and record timing.
    
    Returns:
      predictions: list of normalized label strings ('POSITIVE'/'NEGATIVE')
      confidences: list of float prediction scores
      total_time_sec: total inference execution time in seconds
      avg_latency_ms: average latency per sample in milliseconds
    """
    classifier = get_model_pipeline(model_id, offline_only=offline_only, device=device)

    print(f"[INFO] Running inference on {len(texts)} samples using '{model_id}'...")
    start_time = time.perf_counter()
    raw_results = classifier(texts)
    total_time_sec = time.perf_counter() - start_time

    avg_latency_ms = (total_time_sec / len(texts)) * 1000.0

    predictions = []
    confidences = []
    for item in raw_results:
        predictions.append(normalize_label(item["label"]))
        confidences.append(round(float(item["score"]), 4))

    return predictions, confidences, round(total_time_sec, 4), round(avg_latency_ms, 2)


# -----------------------------------------------------------------------------
# 6. Metric Calculation
# -----------------------------------------------------------------------------
def compute_metrics(
    y_true: list[str], y_pred: list[str], total_time: float, avg_latency: float
) -> dict:
    """Calculate Accuracy, Macro-F1, and timing metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "total_inference_s": round(float(total_time), 4),
        "avg_latency_ms": round(float(avg_latency), 2),
        "eval_samples": len(y_true),
    }


# -----------------------------------------------------------------------------
# 7. MLflow Run Logging Functions
# -----------------------------------------------------------------------------
def log_model_run(
    run_name: str,
    model_id: str,
    role: str,
    metrics: dict,
    eval_samples: int,
    device: int,
    predictions_path: Path,
    mode: str,
    dataset_name: str = DATASET_NAME,
) -> str:
    """
    Log an individual candidate model evaluation run to MLflow.
    
    Logs:
      - Parameters: hf_model_id, task, role, eval_samples, seed, device, max_length, batch_size, truncation
      - Metrics: accuracy, macro_f1, avg_latency_ms, total_inference_s
      - Tags: student_usn, dataset_name, dataset_source, evaluation_phase, model_role, library versions
      - Artifacts: predictions CSV
    """
    device_str = "cpu" if device == -1 else f"cuda:{device}"

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[INFO] Started MLflow run '{run_name}' (Run ID: {run_id})")

        # 1. Log Parameters (actual settings used)
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": TASK_NAME,
            "role": role,
            "eval_samples": eval_samples,
            "seed": RANDOM_SEED,
            "device": device_str,
            "max_length": MAX_LENGTH,
            "batch_size": eval_samples,
            "truncation": True,
        })

        # 2. Log Metrics (directly from evaluation)
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "avg_latency_ms": metrics["avg_latency_ms"],
            "total_inference_s": metrics["total_inference_s"],
        })

        # 3. Log Tags (programmatically derived)
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "dataset_name": dataset_name,
            "dataset_source": DATASET_SOURCE,
            "dataset_version": DATASET_VERSION,
            "evaluation_phase": mode,
            "model_role": role,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
        })

        # 4. Log Artifacts
        if predictions_path.exists():
            mlflow.log_artifact(str(predictions_path), artifact_path="predictions")
            print(f"[INFO] Logged artifact '{predictions_path.name}' to MLflow run '{run_name}'")

        print(f"[INFO] Completed MLflow run '{run_name}'")
        return run_id


def log_comparison_run(
    metrics_a: dict,
    metrics_b: dict,
    run_id_a: str,
    run_id_b: str,
    summary_path: Path,
    plot_path: Path,
    mode: str,
    eval_samples: int,
    dataset_name: str = DATASET_NAME,
) -> tuple[str, str, float]:
    """
    Log the comparison and decision summary run to MLflow.
    
    Logs:
      - Parameters: primary_metric, margin, direction, tie_breaker, final_eval_samples
      - Metrics: model_A_macro_f1, model_B_macro_f1, delta_primary_metric, accuracy & latency metrics
      - Tags: student_usn, run_type, evaluation_phase, model_A_run_id, model_B_run_id, decision_rule, decision_outcome
      - Artifacts: metrics_summary.json, comparison plot
    """
    delta_primary_metric = round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4)

    # Apply predefined Champion-Challenger decision rule
    if delta_primary_metric >= IMPROVEMENT_MARGIN:
        decision_outcome = "Challenger wins"
        decision_reason = f"Challenger macro_f1 ({metrics_b['macro_f1']}) >= Champion ({metrics_a['macro_f1']}) + {IMPROVEMENT_MARGIN}"
    elif abs(delta_primary_metric) < IMPROVEMENT_MARGIN:
        # Tie-breaker: lower average latency
        if metrics_b["avg_latency_ms"] < metrics_a["avg_latency_ms"]:
            decision_outcome = "Challenger wins"
            decision_reason = (
                f"Macro-F1 within margin (|{delta_primary_metric}| < {IMPROVEMENT_MARGIN}); "
                f"Challenger latency ({metrics_b['avg_latency_ms']}ms) < Champion ({metrics_a['avg_latency_ms']}ms)"
            )
        else:
            decision_outcome = "Champion retained"
            decision_reason = (
                f"Macro-F1 within margin (|{delta_primary_metric}| < {IMPROVEMENT_MARGIN}); "
                f"Champion latency ({metrics_a['avg_latency_ms']}ms) <= Challenger ({metrics_b['avg_latency_ms']}ms)"
            )
    else:
        decision_outcome = "Champion retained"
        decision_reason = f"Challenger macro_f1 ({metrics_b['macro_f1']}) did not exceed Champion ({metrics_a['macro_f1']}) by {IMPROVEMENT_MARGIN}"

    with mlflow.start_run(run_name=COMPARISON_RUN_NAME) as run:
        comparison_run_id = run.info.run_id
        print(f"[INFO] Started MLflow comparison run '{COMPARISON_RUN_NAME}' (Run ID: {comparison_run_id})")

        # 1. Log Decision Parameters
        mlflow.log_params({
            "primary_metric": PRIMARY_METRIC,
            "margin": IMPROVEMENT_MARGIN,
            "direction": METRIC_DIRECTION,
            "tie_breaker": TIE_BREAKER,
            "final_eval_samples": eval_samples,
            "model_A_name": MODEL_A_ID,
            "model_B_name": MODEL_B_ID,
        })

        # 2. Log Comparison Metrics
        mlflow.log_metrics({
            "model_A_macro_f1": metrics_a["macro_f1"],
            "model_B_macro_f1": metrics_b["macro_f1"],
            "delta_primary_metric": delta_primary_metric,
            "model_A_accuracy": metrics_a["accuracy"],
            "model_B_accuracy": metrics_b["accuracy"],
            "model_A_avg_latency_ms": metrics_a["avg_latency_ms"],
            "model_B_avg_latency_ms": metrics_b["avg_latency_ms"],
        })

        # 3. Log Comparison Tags
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "dataset_name": dataset_name,
            "dataset_source": DATASET_SOURCE,
            "dataset_version": DATASET_VERSION,
            "evaluation_phase": mode,
            "run_type": "comparison",
            "model_A_run_id": run_id_a,
            "model_B_run_id": run_id_b,
            "decision_rule": (
                "Challenger wins if delta >= 0.01; else Champion retained; "
                "tie-breaker: lower latency"
            ),
            "decision_outcome": decision_outcome,
            "decision_reason": decision_reason,
        })

        # 4. Log Artifacts
        if summary_path.exists():
            mlflow.log_artifact(str(summary_path), artifact_path="summary")
        if plot_path.exists():
            mlflow.log_artifact(str(plot_path), artifact_path="plots")

        print(f"[INFO] Completed MLflow comparison run '{COMPARISON_RUN_NAME}'")
        return comparison_run_id, decision_outcome, delta_primary_metric


# -----------------------------------------------------------------------------
# 8. Visualization & Reporting
# -----------------------------------------------------------------------------
def generate_comparison_plot(
    metrics_a: dict, metrics_b: dict, output_plot_path: Path, mode: str
) -> None:
    """Generate a side-by-side bar chart comparing Accuracy and Macro-F1."""
    categories = ["Accuracy", "Macro-F1"]
    values_a = [metrics_a["accuracy"], metrics_a["macro_f1"]]
    values_b = [metrics_b["accuracy"], metrics_b["macro_f1"]]

    x = np.arange(len(categories))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(
        x - bar_width / 2,
        values_a,
        bar_width,
        label="Model A (DistilBERT - Champion)",
        color="#3498db",
    )
    rects2 = ax.bar(
        x + bar_width / 2,
        values_b,
        bar_width,
        label="Model B (RoBERTa - Challenger)",
        color="#e74c3c",
    )

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title(
        f"Champion vs. Challenger: Final Performance Comparison [{mode.upper()} BENCHMARK]",
        fontsize=12,
        pad=12,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Comparison plot saved to: {output_plot_path}")


def print_terminal_summary(
    metrics_a: dict,
    metrics_b: dict,
    mode: str,
    run_id_a: str,
    run_id_b: str,
    run_id_comp: str,
    decision_outcome: str,
    delta: float,
) -> None:
    """Print an aligned, professional summary report in the terminal."""
    header = "=" * 84
    print("\n" + header)
    print(f"        FINAL BENCHMARK EVALUATION & MLFLOW EXPERIMENT SUMMARY [{mode.upper()}]")
    print(header)
    print(
        f"{'Metric / Property':<30} | {'Model A (DistilBERT)':<24} | {'Model B (RoBERTa)':<24}"
    )
    print("-" * 84)
    print(f"{'Role':<30} | {MODEL_A_ROLE:<24} | {MODEL_B_ROLE:<24}")
    print(f"{'MLflow Run Name':<30} | {MODEL_A_RUN_NAME:<24} | {MODEL_B_RUN_NAME:<24}")
    print(f"{'MLflow Run ID':<30} | {run_id_a[:12]}...{'':<12} | {run_id_b[:12]}...{'':<12}")
    print(f"{'Samples Evaluated':<30} | {metrics_a['eval_samples']:<24} | {metrics_b['eval_samples']:<24}")
    print(f"{'Accuracy':<30} | {metrics_a['accuracy']:<24.4f} | {metrics_b['accuracy']:<24.4f}")
    print(f"{'Macro-F1 (Primary Metric)':<30} | {metrics_a['macro_f1']:<24.4f} | {metrics_b['macro_f1']:<24.4f}")
    print(f"{'Average Latency (ms/sample)':<30} | {metrics_a['avg_latency_ms']:<24.2f} | {metrics_b['avg_latency_ms']:<24.2f}")
    print(f"{'Total Inference Time (s)':<30} | {metrics_a['total_inference_s']:<24.4f} | {metrics_b['total_inference_s']:<24.4f}")
    print("-" * 84)
    print(f"{'Comparison Run Name':<30} | {COMPARISON_RUN_NAME:<51}")
    print(f"{'Comparison Run ID':<30} | {run_id_comp:<51}")
    print(f"{'Delta Primary Metric (F1)':<30} | {delta:<51.4f}")
    print(f"{'Required Margin':<30} | +{IMPROVEMENT_MARGIN:<50.2f}")
    print(f"{'Decision Outcome':<30} | {decision_outcome:<51}")
    print(header)


# -----------------------------------------------------------------------------
# 9. Main Pipeline Entrypoint
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Champion-Challenger evaluation with MLflow tracking."
    )
    parser.add_argument(
        "--mode",
        choices=["dev", "full"],
        default="full",
        help="Execution mode: 'full' runs full final benchmark (100 samples); 'dev' for fast debug.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Inference device: -1 for CPU, 0+ for GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)

    # Both modes safely use local cache first
    offline_only = True

    project_root = Path(__file__).resolve().parent
    eval_csv_path = project_root / "data" / "eval_inputs.csv"
    outputs_dir = project_root / "outputs"
    plots_dir = outputs_dir / "plots"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Starting evaluation pipeline in [{args.mode.upper()}] mode...")
    print(f"[INFO] Offline-only caching protection: {offline_only}")

    # 1. Verify MLflow Tracking Server Connectivity & Set Experiment
    exp_id, mlflow_client = check_and_configure_mlflow(
        tracking_uri=MLFLOW_TRACKING_URI,
        experiment_name=MLFLOW_EXPERIMENT_NAME,
    )

    # 2. Load and Validate Dataset
    print(f"[INFO] Loading evaluation dataset from: {eval_csv_path}")
    eval_df = load_and_validate_dataset(eval_csv_path)
    print(f"[INFO] Dataset validation passed: {len(eval_df)} samples ready.")

    texts = eval_df["text"].tolist()
    ground_truth = eval_df["ground_truth"].tolist()

    # 3. Evaluate Model A (DistilBERT - Initial Champion)
    print(f"\n--- Evaluating Model A ({MODEL_A_ID}) ---")
    preds_a, confs_a, time_a, lat_a = run_model_inference(
        MODEL_A_ID, texts, offline_only=offline_only, device=args.device
    )
    metrics_a = compute_metrics(ground_truth, preds_a, time_a, lat_a)

    preds_a_path = outputs_dir / "predictions_A.csv"
    pd.DataFrame({
        "id": eval_df["id"],
        "text": eval_df["text"],
        "ground_truth": ground_truth,
        "prediction": preds_a,
        "confidence": confs_a,
        "is_correct": [p == g for p, g in zip(preds_a, ground_truth)],
        "model_id": MODEL_A_ID,
    }).to_csv(preds_a_path, index=False)
    print(f"[INFO] Model A predictions saved to: {preds_a_path}")

    # 4. Evaluate Model B (RoBERTa-large - Challenger)
    print(f"\n--- Evaluating Model B ({MODEL_B_ID}) ---")
    preds_b, confs_b, time_b, lat_b = run_model_inference(
        MODEL_B_ID, texts, offline_only=offline_only, device=args.device
    )
    metrics_b = compute_metrics(ground_truth, preds_b, time_b, lat_b)

    preds_b_path = outputs_dir / "predictions_B.csv"
    pd.DataFrame({
        "id": eval_df["id"],
        "text": eval_df["text"],
        "ground_truth": ground_truth,
        "prediction": preds_b,
        "confidence": confs_b,
        "is_correct": [p == g for p, g in zip(preds_b, ground_truth)],
        "model_id": MODEL_B_ID,
    }).to_csv(preds_b_path, index=False)
    print(f"[INFO] Model B predictions saved to: {preds_b_path}")

    # 5. Generate Metrics Summary JSON & Plot
    delta_primary = round(metrics_b["macro_f1"] - metrics_a["macro_f1"], 4)
    summary_data = {
        "execution_mode": args.mode,
        "student_usn": STUDENT_USN,
        "evaluation_dataset": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_version": DATASET_VERSION,
        "eval_samples": len(eval_df),
        "random_seed": RANDOM_SEED,
        "mlflow_experiment": MLFLOW_EXPERIMENT_NAME,
        "model_A": {
            "model_id": MODEL_A_ID,
            "role": MODEL_A_ROLE,
            "run_name": MODEL_A_RUN_NAME,
            "metrics": metrics_a,
        },
        "model_B": {
            "model_id": MODEL_B_ID,
            "role": MODEL_B_ROLE,
            "run_name": MODEL_B_RUN_NAME,
            "metrics": metrics_b,
        },
        "comparison": {
            "primary_metric": PRIMARY_METRIC,
            "margin": IMPROVEMENT_MARGIN,
            "direction": METRIC_DIRECTION,
            "tie_breaker": TIE_BREAKER,
            "delta_primary_metric": delta_primary,
            "rule_based_winner": (
                "Challenger" if delta_primary >= IMPROVEMENT_MARGIN
                else ("Challenger" if abs(delta_primary) < IMPROVEMENT_MARGIN and metrics_b["avg_latency_ms"] < metrics_a["avg_latency_ms"] else "Champion")
            ),
        },
    }
    summary_path = outputs_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[INFO] Metrics summary saved to: {summary_path}")

    plot_path = plots_dir / "model_comparison.png"
    generate_comparison_plot(metrics_a, metrics_b, plot_path, mode=args.mode)

    # 6. Log MLflow Runs
    print("\n--- Logging MLflow Runs ---")
    run_id_a = log_model_run(
        run_name=MODEL_A_RUN_NAME,
        model_id=MODEL_A_ID,
        role=MODEL_A_ROLE,
        metrics=metrics_a,
        eval_samples=len(eval_df),
        device=args.device,
        predictions_path=preds_a_path,
        mode=args.mode,
    )

    run_id_b = log_model_run(
        run_name=MODEL_B_RUN_NAME,
        model_id=MODEL_B_ID,
        role=MODEL_B_ROLE,
        metrics=metrics_b,
        eval_samples=len(eval_df),
        device=args.device,
        predictions_path=preds_b_path,
        mode=args.mode,
    )

    run_id_comp, decision_outcome, delta = log_comparison_run(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        run_id_a=run_id_a,
        run_id_b=run_id_b,
        summary_path=summary_path,
        plot_path=plot_path,
        mode=args.mode,
        eval_samples=len(eval_df),
    )

    # 7. Print Terminal Summary
    print_terminal_summary(
        metrics_a,
        metrics_b,
        args.mode,
        run_id_a,
        run_id_b,
        run_id_comp,
        decision_outcome,
        delta,
    )


if __name__ == "__main__":
    main()
