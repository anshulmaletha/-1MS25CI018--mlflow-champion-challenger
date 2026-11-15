"""
predict_from_mlflow.py - Decoupled Champion Inference from MLflow Model Registry

Project: Champion–Challenger Model Comparison with Hugging Face and MLflow
Student USN: 1MS25CI018
Phase: Phase 5 - Champion Prediction on Unseen Data

Objective:
1. Connect to the MLflow Tracking Server & Model Registry.
2. Dynamically resolve the '@champion' alias under 'HF_Sentiment_Champion_Challenger'.
3. Discover the resolved model version and underlying metadata from the registry.
4. Load the winning Champion model strictly through its MLflow Model URI:
     models:/HF_Sentiment_Champion_Challenger@champion
5. Ingest completely unseen, unlabeled examples from 'data/new_data.csv'.
6. Execute inference using the MLflow PyFunc interface (zero direct framework coupling).
7. Save predictions to 'outputs/new_data_predictions.csv'.

Strictly compliant:
- Only MLflow PyFunc is used for model loading and prediction.
- Dynamic Champion alias resolution via MLflow Model Registry.
- No hard-coded versions or run IDs.
"""

from pathlib import Path
import sys
import urllib.error
import urllib.request

# Ensure UTF-8 console encoding on Windows
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Configuration Constants
# -----------------------------------------------------------------------------
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REGISTERED_MODEL_NAME = "HF_Sentiment_Champion_Challenger"
CHAMPION_ALIAS = "champion"
CHAMPION_MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"


# -----------------------------------------------------------------------------
# 2. Server Connectivity & Dynamic Alias Resolution
# -----------------------------------------------------------------------------
def verify_server_connection(tracking_uri: str) -> MlflowClient:
    """Verify that the MLflow tracking server is live and reachable."""
    health_endpoint = f"{tracking_uri.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_endpoint, timeout=5) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"MLflow server responded with non-200 status code: {response.status}"
                )
    except Exception as err:
        raise ConnectionError(
            f"\n[ERROR] Unable to connect to MLflow tracking server at '{tracking_uri}'.\n"
            "Please ensure the MLflow tracking server is running on port 5000:\n"
            "  .\\.venv\\Scripts\\python -m mlflow server --host 127.0.0.1 --port 5000 "
            "--backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns\n"
            f"Underlying connection error: {err}"
        ) from err

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def resolve_champion_metadata(
    client: MlflowClient, model_name: str, alias: str
) -> dict:
    """
    Dynamically resolve the champion model version and metadata from MLflow.
    Guarantees no version numbers or model IDs are hard-coded in the prediction code.
    """
    try:
        model_version = client.get_model_version_by_alias(model_name, alias)
    except Exception as err:
        raise RuntimeError(
            f"Failed to resolve alias '@{alias}' for registered model '{model_name}'.\n"
            "Ensure 'register_champion.py' has been executed successfully.\n"
            f"Underlying error: {err}"
        ) from err

    tags = model_version.tags or {}
    return {
        "registered_model_name": model_name,
        "alias": alias,
        "version": str(model_version.version),
        "source_run_id": model_version.run_id,
        "hf_model_id": tags.get("hf_model_id", "Unknown"),
        "task": tags.get("task", "sentiment-classification"),
        "role": tags.get("role", "Champion"),
        "accuracy": tags.get("accuracy", "N/A"),
        "macro_f1": tags.get("macro_f1", "N/A"),
        "avg_latency_ms": tags.get("avg_latency_ms", "N/A"),
        "description": model_version.description or "",
    }


# -----------------------------------------------------------------------------
# 3. Data Validation
# -----------------------------------------------------------------------------
def load_and_validate_new_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate unseen production input samples.
    
    Validation Checks:
      1. File existence.
      2. Required columns ('id', 'text') present.
      3. No ground-truth label column (purely unlabeled production data).
      4. Non-empty text entries.
      5. Strictly unique IDs.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"New data file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate schema
    required_cols = {"id", "text"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Dataset missing required columns {required_cols}. Found: {df.columns}")

    if "ground_truth" in df.columns or "label" in df.columns:
        raise ValueError(
            "New data file must NOT contain ground-truth labels. It must represent unseen production data."
        )

    # Check non-empty text
    df["text"] = df["text"].astype(str).str.strip()
    if (df["text"] == "").any():
        raise ValueError("New data contains empty or whitespace-only text entries.")

    if not df["id"].is_unique:
        raise ValueError("New data 'id' column contains duplicate IDs.")

    return df


# -----------------------------------------------------------------------------
# 4. Main Inference Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 82)
    print("      CHAMPION MODEL PRODUCTION INFERENCE VIA MLFLOW (PHASE 5)")
    print("=" * 82)

    # 1. Connect to MLflow Server
    print(f"[INFO] Connecting to MLflow tracking server: {MLFLOW_TRACKING_URI}")
    client = verify_server_connection(MLFLOW_TRACKING_URI)

    # 2. Dynamically Resolve Champion Alias
    print(f"[INFO] Resolving Champion alias '@{CHAMPION_ALIAS}' under '{REGISTERED_MODEL_NAME}'...")
    meta = resolve_champion_metadata(client, REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    print("\n--- Dynamically Resolved Champion Metadata ---")
    print(f"  Registered Model:    {meta['registered_model_name']}")
    print(f"  Alias:               @{meta['alias']}")
    print(f"  Resolved Version:    Version {meta['version']}")
    print(f"  Underlying HF Model: {meta['hf_model_id']}")
    print(f"  Benchmark Macro-F1:  {meta['macro_f1']}")
    print(f"  Benchmark Accuracy:  {meta['accuracy']}")
    print(f"  Avg Latency:         {meta['avg_latency_ms']} ms/sample")
    print(f"  Source MLflow Run:   {meta['source_run_id']}")

    # 3. Load Model via MLflow Model URI (No Direct Framework Calls)
    print(f"\n[INFO] Loading Champion model via MLflow URI: '{CHAMPION_MODEL_URI}'...")
    try:
        champion_model = mlflow.pyfunc.load_model(CHAMPION_MODEL_URI)
        print("[SUCCESS] Champion model loaded successfully through MLflow PyFunc!")
    except Exception as err:
        raise RuntimeError(
            f"Failed to load Champion model from URI '{CHAMPION_MODEL_URI}'.\n"
            f"Underlying error: {err}"
        ) from err

    # 4. Ingest Brand-New Unseen Data
    project_root = Path(__file__).resolve().parent
    new_data_path = project_root / "data" / "new_data.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading unseen test data from: {new_data_path}")
    new_df = load_and_validate_new_data(new_data_path)
    print(f"[INFO] Loaded {len(new_df)} unseen test samples.")

    # 5. Execute Prediction via MLflow PyFunc
    print(f"[INFO] Running decoupled inference using MLflow Champion...")
    predictions_df = champion_model.predict(new_df[["text"]])

    # 6. Assemble and Verify Predictions
    out_df = pd.DataFrame({
        "id": new_df["id"],
        "text": new_df["text"],
        "prediction": predictions_df["prediction"],
        "confidence": predictions_df["confidence"],
    })

    # Integrity Assertions
    assert len(out_df) == len(new_df), "Output row count does not match input row count!"
    assert list(out_df["id"]) == list(new_df["id"]), "Output IDs do not match input IDs!"
    assert out_df["prediction"].str.strip().ne("").all(), "Found empty predictions!"

    # 7. Save Output CSV
    out_csv_path = outputs_dir / "new_data_predictions.csv"
    out_df.to_csv(out_csv_path, index=False)
    print(f"[INFO] Predictions successfully saved to: {out_csv_path}")

    # 8. Display Results in Aligned Console Table
    header = "=" * 90
    print("\n" + header)
    print("                     BRAND-NEW DATA CHAMPION PREDICTION RESULTS")
    print(header)
    print(f"Champion Model URI:   {CHAMPION_MODEL_URI}")
    print(f"Resolved Version:     Version {meta['version']} ({meta['hf_model_id']})")
    print(f"Total Samples:        {len(out_df)}")
    print("-" * 90)
    for _, row in out_df.iterrows():
        print(f"Sample {row['id']}: \"{row['text']}\"")
        print(f"  Prediction: {row['prediction']:<10} | Confidence Score: {row['confidence']:.4f}")
        print("-" * 90)

    print(f"[SUCCESS] Phase 5 completed: {len(out_df)} predictions generated via MLflow Champion.")
    print("=" * 90)


if __name__ == "__main__":
    main()
