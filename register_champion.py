"""
register_champion.py - Model Registration & Champion/Challenger Aliasing

Project: Champion–Challenger Model Comparison with Hugging Face and MLflow
Student USN: 1MS25CI018
Phase: Phase 4 - Model Registry & Alias Governance

Objective:
1. Locate the final benchmark evaluation runs for Model A and Model B in MLflow.
2. Package both models into independently loadable MLflow pyfunc models with full metadata.
3. Register BOTH candidate models under exactly ONE registered model name:
     'HF_Sentiment_Champion_Challenger'
4. Assign governance aliases according to the predefined decision rule:
     @champion   -> Model B (siebert/sentiment-roberta-large-english)
     @challenger -> Model A (distilbert-base-uncased-finetuned-sst-2-english)
5. Programmatically verify alias resolution and execute smoke inference directly
   through the registered MLflow Model URIs:
     models:/HF_Sentiment_Champion_Challenger@champion
     models:/HF_Sentiment_Champion_Challenger@challenger
6. Update the MLflow comparison run metadata with the finalized governance decision.
"""

import os
from pathlib import Path
import sys
import tempfile
import time
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# -----------------------------------------------------------------------------
# 1. Configuration Constants
# -----------------------------------------------------------------------------
STUDENT_USN = "1MS25CI018"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "HF_Sentiment_Champion_Challenger"

MODEL_A_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B_ID = "siebert/sentiment-roberta-large-english"

# Predefined Decision Rule Parameters
PRIMARY_METRIC = "macro_f1"
IMPROVEMENT_MARGIN = 0.01
METRIC_DIRECTION = "higher_is_better"
TIE_BREAKER = "lower_avg_latency_ms"


# -----------------------------------------------------------------------------
# 2. PyFunc Model Wrapper Definition
# -----------------------------------------------------------------------------
class SentimentPipelineModel(mlflow.pyfunc.PythonModel):
    """
    Production-grade MLflow PyFunc wrapper for Hugging Face Sentiment Analysis.
    
    Self-contained: loads the saved tokenizer and transformer weights directly
    from the packaged MLflow artifact directory. Accepts a pandas DataFrame or list
    of strings and outputs standardized predictions and confidence scores.
    """
    def __init__(self, hf_model_id: str = "", max_length: int = 512):
        self.hf_model_id = hf_model_id
        self.max_length = max_length

    def load_context(self, context):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        model_dir = context.artifacts["model_dir"]
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
        self.pipe = pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=self.max_length,
        )

    def predict(self, context, model_input):
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        elif isinstance(model_input, (list, tuple)):
            texts = [str(t) for t in model_input]
        else:
            texts = [str(model_input)]

        raw_outputs = self.pipe(texts)
        results = []
        for out in raw_outputs:
            raw_lbl = str(out["label"]).strip().upper()
            norm_lbl = (
                "POSITIVE"
                if raw_lbl in ("POSITIVE", "LABEL_1", "POS", "1")
                else "NEGATIVE"
            )
            results.append({
                "prediction": norm_lbl,
                "confidence": round(float(out["score"]), 4),
            })
        return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
def verify_server_connection(tracking_uri: str) -> MlflowClient:
    """Verify MLflow tracking server is reachable."""
    health_endpoint = f"{tracking_uri.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(health_endpoint, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Server responded with HTTP {resp.status}")
    except Exception as err:
        raise ConnectionError(
            f"ERROR: Cannot connect to MLflow tracking server at '{tracking_uri}'.\n"
            f"Please ensure the server is active on port 5000.\nCause: {err}"
        ) from err

    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def find_final_runs(client: MlflowClient, exp_id: str) -> dict:
    """Locate the final benchmark runs for Model A, Model B, and comparison."""
    runs = client.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"],
        max_results=50,
    )

    final_runs = {}
    for r in runs:
        phase = r.data.tags.get("evaluation_phase", "")
        name = r.info.run_name
        if phase == "full" and name not in final_runs:
            final_runs[name] = r

    required = {"A_sentiment_distilbert", "B_sentiment_roberta", "compare_sentiment"}
    missing = required - set(final_runs.keys())
    if missing:
        raise RuntimeError(
            f"Could not locate all final benchmark runs (phase='full'). Missing: {missing}\n"
            "Please ensure 'evaluate.py --mode full' was executed successfully."
        )

    return final_runs


def package_and_log_model(
    model_id: str,
    run_id: str,
    artifact_path: str = "model",
) -> str:
    """
    Export tokenizer and model weights from local cache and log as an MLflow PyFunc model.
    Includes code_paths to ensure independent loadability across all environments.
    """
    this_script_path = str(Path(__file__).resolve())
    print(f"[INFO] Exporting model '{model_id}' for MLflow packaging...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Load directly from local offline cache
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, local_files_only=True
        )

        tokenizer.save_pretrained(tmp_dir)
        model.save_pretrained(tmp_dir)

        print(f"[INFO] Logging MLflow PyFunc model into run '{run_id}'...")
        with mlflow.start_run(run_id=run_id):
            model_info = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=SentimentPipelineModel(hf_model_id=model_id),
                artifacts={"model_dir": tmp_dir},
                code_paths=[this_script_path],
                pip_requirements=["transformers", "torch", "pandas"],
            )

    model_uri = f"runs:/{run_id}/{artifact_path}"
    print(f"[INFO] Model logged successfully. URI: {model_uri}")
    return model_uri


# -----------------------------------------------------------------------------
# 4. Main Registration & Aliasing Workflow
# -----------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("      MLFLOW MODEL REGISTRY & CHAMPION/CHALLENGER ALIASING (PHASE 4)")
    print("=" * 80)

    # 1. Connect to MLflow Tracking Server
    print(f"[INFO] Connecting to MLflow tracking server at: {MLFLOW_TRACKING_URI}")
    client = verify_server_connection(MLFLOW_TRACKING_URI)

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{MLFLOW_EXPERIMENT_NAME}' not found on server.")
    print(f"[INFO] Found experiment '{MLFLOW_EXPERIMENT_NAME}' (ID: {exp.experiment_id})")

    # 2. Locate Final Benchmark Runs
    final_runs = find_final_runs(client, exp.experiment_id)
    run_a = final_runs["A_sentiment_distilbert"]
    run_b = final_runs["B_sentiment_roberta"]
    run_comp = final_runs["compare_sentiment"]

    print("\n--- Final Benchmark Runs Located ---")
    print(f"  Model A Run:   {run_a.info.run_id} ({run_a.info.run_name})")
    print(f"  Model B Run:   {run_b.info.run_id} ({run_b.info.run_name})")
    print(f"  Comparison Run: {run_comp.info.run_id} ({run_comp.info.run_name})")

    f1_a = float(run_a.data.metrics["macro_f1"])
    f1_b = float(run_b.data.metrics["macro_f1"])
    delta_f1 = round(f1_b - f1_a, 4)
    print(f"\n[INFO] Model A Macro-F1: {f1_a:.4f}")
    print(f"[INFO] Model B Macro-F1: {f1_b:.4f}")
    print(f"[INFO] Delta Primary Metric: {delta_f1:+.4f} (Required Margin: +{IMPROVEMENT_MARGIN:.2f})")

    # 3. Validate Decision Rule
    if delta_f1 >= IMPROVEMENT_MARGIN:
        champion_model_id = MODEL_B_ID
        challenger_model_id = MODEL_A_ID
        winner_label = "Model B"
        print(f"[INFO] Decision Rule Result: CHALLENGER WINS ({winner_label} promoted to @champion)")
    else:
        champion_model_id = MODEL_A_ID
        challenger_model_id = MODEL_B_ID
        winner_label = "Model A"
        print(f"[INFO] Decision Rule Result: CHAMPION RETAINED ({winner_label} retained as @champion)")

    # 4. Package and Register Model A
    print(f"\n--- Packaging and Registering Model A ({MODEL_A_ID}) ---")
    model_a_uri = package_and_log_model(MODEL_A_ID, run_a.info.run_id)
    mv_a = mlflow.register_model(model_uri=model_a_uri, name=REGISTERED_MODEL_NAME)
    print(f"[INFO] Model A registered as Version {mv_a.version}")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv_a.version,
        description="Candidate Model A: DistilBERT fine-tuned on SST-2 (Final Role: Challenger)",
    )
    tags_a = {
        "hf_model_id": MODEL_A_ID,
        "task": "sentiment-classification",
        "student_usn": STUDENT_USN,
        "role": "Challenger",  # Post-decision final role
        "source_run_id": run_a.info.run_id,
        "accuracy": str(run_a.data.metrics["accuracy"]),
        "macro_f1": str(f1_a),
        "avg_latency_ms": str(run_a.data.metrics["avg_latency_ms"]),
    }
    for k, v in tags_a.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv_a.version, k, v)

    # 5. Package and Register Model B
    print(f"\n--- Packaging and Registering Model B ({MODEL_B_ID}) ---")
    model_b_uri = package_and_log_model(MODEL_B_ID, run_b.info.run_id)
    mv_b = mlflow.register_model(model_uri=model_b_uri, name=REGISTERED_MODEL_NAME)
    print(f"[INFO] Model B registered as Version {mv_b.version}")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv_b.version,
        description="Candidate Model B: RoBERTa-large fine-tuned on diverse sentiment (Final Role: Champion)",
    )
    tags_b = {
        "hf_model_id": MODEL_B_ID,
        "task": "sentiment-classification",
        "student_usn": STUDENT_USN,
        "role": "Champion",  # Post-decision final role
        "source_run_id": run_b.info.run_id,
        "accuracy": str(run_b.data.metrics["accuracy"]),
        "macro_f1": str(f1_b),
        "avg_latency_ms": str(run_b.data.metrics["avg_latency_ms"]),
    }
    for k, v in tags_b.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, mv_b.version, k, v)

    # 6. Assign Aliases Strictly per Predefined Decision Rule
    print("\n--- Assigning MLflow Model Registry Aliases ---")
    # Model B won -> @champion
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=mv_b.version,
    )
    print(f"[INFO] Assigned alias '@champion'   -> Version {mv_b.version} ({MODEL_B_ID})")

    # Model A lost -> @challenger
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="challenger",
        version=mv_a.version,
    )
    print(f"[INFO] Assigned alias '@challenger' -> Version {mv_a.version} ({MODEL_A_ID})")

    # 7. Programmatically Verify Alias Resolution
    print("\n--- Verifying Alias Resolution from MLflow Registry ---")
    champ_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
    chall_mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "challenger")

    print(f"  Alias '@champion'   resolves to Version {champ_mv.version}")
    print(f"    Underlying HF Model: {champ_mv.tags.get('hf_model_id')}")
    print(f"    Macro-F1:            {champ_mv.tags.get('macro_f1')}")

    print(f"  Alias '@challenger' resolves to Version {chall_mv.version}")
    print(f"    Underlying HF Model: {chall_mv.tags.get('hf_model_id')}")
    print(f"    Macro-F1:            {chall_mv.tags.get('macro_f1')}")

    assert champ_mv.tags.get("hf_model_id") == MODEL_B_ID, "Champion alias does NOT map to Model B!"
    assert chall_mv.tags.get("hf_model_id") == MODEL_A_ID, "Challenger alias does NOT map to Model A!"
    print("[SUCCESS] Alias resolution strictly verified!")

    # 8. Smoke Test Both Registered Models via MLflow URIs
    print("\n--- Smoke Testing Inference via MLflow Model URIs ---")
    champion_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    challenger_uri = f"models:/{REGISTERED_MODEL_NAME}@challenger"

    print(f"[INFO] Loading Champion from: {champion_uri}")
    champion_pyfunc = mlflow.pyfunc.load_model(champion_uri)

    print(f"[INFO] Loading Challenger from: {challenger_uri}")
    challenger_pyfunc = mlflow.pyfunc.load_model(challenger_uri)

    test_samples = pd.DataFrame({
        "text": [
            "The customer service was exceptional and resolved my concern immediately.",
            "The software is buggy, crashes frequently, and ruined my entire workflow.",
            "Although the design is minimalist, the usability and responsiveness are outstanding.",
        ]
    })

    print(f"[INFO] Running Champion predictions on {len(test_samples)} test samples...")
    champ_preds = champion_pyfunc.predict(test_samples)

    print(f"[INFO] Running Challenger predictions on {len(test_samples)} test samples...")
    chall_preds = challenger_pyfunc.predict(test_samples)

    smoke_results = pd.DataFrame({
        "Input Text": test_samples["text"],
        "Champion (RoBERTa) Pred": champ_preds["prediction"],
        "Champion Conf": champ_preds["confidence"],
        "Challenger (DistilBERT) Pred": chall_preds["prediction"],
        "Challenger Conf": chall_preds["confidence"],
    })
    print("\n" + "=" * 95)
    print("                   REGISTERED MODEL SMOKE TEST RESULTS")
    print("=" * 95)
    for idx, row in smoke_results.iterrows():
        print(f"Sample {idx+1}: \"{row['Input Text']}\"")
        print(f"  Champion   (@champion):   {row['Champion (RoBERTa) Pred']} (Confidence: {row['Champion Conf']})")
        print(f"  Challenger (@challenger): {row['Challenger (DistilBERT) Pred']} (Confidence: {row['Challenger Conf']})")
        print("-" * 95)

    # 9. Update Comparison Run Metadata
    print("\n--- Updating Comparison Run Metadata in MLflow ---")
    comp_run_id = run_comp.info.run_id
    client.set_tag(comp_run_id, "primary_metric", PRIMARY_METRIC)
    client.set_tag(comp_run_id, "margin", str(IMPROVEMENT_MARGIN))
    client.set_tag(comp_run_id, "direction", METRIC_DIRECTION)
    client.set_tag(comp_run_id, "delta_primary_metric", str(delta_f1))
    client.set_tag(comp_run_id, "winner", winner_label)
    client.set_tag(comp_run_id, "winner_hf_model_id", champion_model_id)
    client.set_tag(comp_run_id, "champion_model_id", champion_model_id)
    client.set_tag(comp_run_id, "challenger_model_id", challenger_model_id)
    client.set_tag(comp_run_id, "registered_model_name", REGISTERED_MODEL_NAME)
    client.set_tag(comp_run_id, "champion_version", str(mv_b.version))
    client.set_tag(comp_run_id, "challenger_version", str(mv_a.version))
    client.set_tag(comp_run_id, "champion_uri", champion_uri)
    client.set_tag(comp_run_id, "challenger_uri", challenger_uri)
    client.set_tag(comp_run_id, "governance_status", "PROMOTED_AND_ALIASED")
    print(f"[INFO] Updated comparison run '{comp_run_id}' with final governance metadata.")

    print("\n" + "=" * 80)
    print("      PHASE 4 REGISTRY & ALIASING SUCCESSFULLY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
