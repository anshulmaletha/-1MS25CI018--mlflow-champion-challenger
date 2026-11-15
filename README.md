# Champion–Challenger Model Comparison with Hugging Face and MLflow

**Student USN:** 1MS25CI018  
**Project Type:** College Mini-Project / MLOps Pipeline Demonstration  
**Domain:** Applied Machine Learning & MLOps (Experiment Tracking & Model Governance)

---

## 1. Project Objective

In production machine learning systems, deploying new models requires empirical evidence that a prospective candidate outperforms the incumbent baseline under identical, controlled testing conditions. This project implements an automated, reproducible **Champion–Challenger** evaluation workflow:

1. **Controlled Evaluation:** Two pre-trained Hugging Face sentiment analysis models are evaluated on an identical test benchmark (`data/eval_inputs.csv`) under consistent inference configurations.
2. **Experiment Tracking:** MLflow tracks parameters, evaluation metrics (Accuracy, Macro-F1, Latency, Total Runtime), and artifacts for each candidate model and generates a comparison summary run.
3. **Automated Governance & Champion Selection:** An explicit, pre-defined decision rule selects the Champion based on Macro-F1 and inference latency.
4. **Model Registry & Aliasing:** Both candidate models are packaged and registered in the MLflow Model Registry under a unified model name (`HF_Sentiment_Champion_Challenger`), with role-based aliases (`@champion` and `@challenger`) dynamically assigned based on empirical results.
5. **Decoupled Production Inference:** Completely unseen samples (`data/new_data.csv`) are predicted by loading the winning Champion model directly through its MLflow Model URI (`models:/HF_Sentiment_Champion_Challenger@champion`), ensuring zero direct pipeline coupling.

---

## 2. Selected Task & Candidate Models

- **Task:** Sentiment Classification (Binary: Positive vs. Negative)
- **Model A (Baseline / Initial Champion):**
  - **Hugging Face ID:** `distilbert-base-uncased-finetuned-sst-2-english`
  - **Architecture:** DistilBERT (lightweight, 6 layers, 66M parameters)
  - **Role:** Initial Champion benchmark focusing on high inference speed and low resource footprint.
- **Model B (Challenger):**
  - **Hugging Face ID:** `siebert/sentiment-roberta-large-english`
  - **Architecture:** RoBERTa-large (deep transformer, 24 layers, ~355M parameters)
  - **Role:** Challenger model evaluated to assess whether its higher representational capacity justifies its latency trade-off.

---

## 3. Evaluation Principle & Data Separation

To ensure absolute fairness, transparency, and reproducibility:
- **Same Inputs:** Both models evaluate the exact same set of test examples in the exact same sequence.
- **Strict Data Segregation:**
  - `data/eval_inputs.csv`: Contains the ground-truth labeled evaluation samples used exclusively for model comparison and metric calculation.
  - `data/new_data.csv`: Contains unlabeled real-world examples reserved strictly for post-registration inference with the finalized Champion model. New examples are strictly excluded from the evaluation set.

### Benchmark Dataset Definitions

| Benchmark Tier | Sample Count | Source & Methodology | Purpose |
| :--- | :--- | :--- | :--- |
| **Development Benchmark** | 10 samples | Curated balanced examples (5 positive, 5 negative) | Used in Phase 2/3 for rapid pipeline verification, error trapping, and offline caching tests without heavy compute. |
| **Final Evaluation Benchmark** | 100 samples | Extracted deterministically (`seed=42`) from the official **Stanford Sentiment Treebank (SST-2)** validation split (`stanfordnlp/sst2`), strictly balanced (50 positive, 50 negative). | Used for the official Champion–Challenger empirical comparison and governance decision. |

---

## 4. Predefined Champion–Challenger Decision Rule

The decision policy is established **prior to running experiments** to prevent confirmation bias or post-hoc threshold adjustment:

- **Primary Metric:** `macro_f1` (Macro-averaged F1 Score)
- **Optimization Direction:** Higher is better
- **Minimum Improvement Margin ($\Delta$):** `0.01` (1.0 percentage points)
- **Decision Logic:**
  - If $\text{Macro-F1}_{\text{Challenger}} \ge \text{Macro-F1}_{\text{Champion}} + 0.01$:  
    **Challenger** is declared the winner and promoted to `@champion`.
  - Otherwise ($\text{Macro-F1}_{\text{Challenger}} < \text{Macro-F1}_{\text{Champion}} + 0.01$):  
    **Model A (Champion)** retains its title.
  - **Tie-Breaker:** If the Macro-F1 difference is within the threshold margin ($|\Delta\text{F1}| < 0.01$), the model with the **lower average inference latency** (milliseconds per sample) wins.

---

## 5. Final Benchmark Experimental Results

Evaluated on the 100-sample SST-2 validation benchmark (`data/eval_inputs.csv`):

| Metric / Property | Model A (DistilBERT) | Model B (RoBERTa-large) | Differential ($\Delta = \text{B} - \text{A}$) |
| :--- | :--- | :--- | :--- |
| **Initial Role** | Champion (Initial Benchmark) | Challenger | — |
| **MLflow Run Name** | `A_sentiment_distilbert` | `B_sentiment_roberta` | `compare_sentiment` |
| **MLflow Run ID** | `fd7daceb5bdd467590e8c83f9691445b` | `f94e2a6dc9d043d2aba5bf7debc08e54` | `29c43b2f739f4bf5aae1b6b7029b0c2d` |
| **Samples Evaluated** | 100 | 100 | — |
| **Accuracy** | **0.9200** (92.0%) | **0.9400** (94.0%) | **+0.0200** (+2.0%) |
| **Macro-F1** (Primary) | **0.9200** | **0.9399** | **+0.0199** (+1.99%) |
| **Average Latency** | **113.42 ms/sample** | **691.70 ms/sample** | +578.28 ms/sample (6.1x slower) |
| **Total Inference Time**| 11.34 s | 69.17 s | — |
| **Decision Outcome** | Incumbent | **Challenger wins** | $\Delta\text{F1} = +0.0199 \ge 0.01$ |

### Decision Summary

- **Final Champion:** Model B (`siebert/sentiment-roberta-large-english`), Macro-F1: `0.9399`
- **Final Challenger:** Model A (`distilbert-base-uncased-finetuned-sst-2-english`), Macro-F1: `0.9200`
- **Empirical Differential:** $\Delta = +0.0199$ (exceeds the required margin of $+0.01$).
- **Outcome:** **Model B became the final Champion.**

---

## 6. MLflow Model Registry & Alias Governance

Both candidate models are packaged as self-contained MLflow PyFunc artifacts and registered under exactly **ONE** registered model name:

- **Registered Model Name:** `HF_Sentiment_Champion_Challenger`
- **Model Registry URI (Tracking Server):** `http://127.0.0.1:5000`

### Registered Versions & Alias Mapping

| Version | Underlying Model ID | Final Role | MLflow Alias | Macro-F1 | Avg Latency | Source Run ID |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | `distilbert-base-uncased-finetuned-sst-2-english` | Challenger | `@challenger` | 0.9200 | 113.42 ms | `fd7daceb5bdd467590e8c83f9691445b` |
| **2** | `siebert/sentiment-roberta-large-english` | Champion | `@champion` | 0.9399 | 691.70 ms | `f94e2a6dc9d043d2aba5bf7debc08e54` |

### Engineering Trade-Off Analysis

> [!NOTE]
> **Performance vs. Latency Trade-Off:**
> While Model B (`siebert/sentiment-roberta-large-english`) achieved higher classification performance (+1.99% Macro-F1 and +2.0% Accuracy), its 24-layer RoBERTa-large architecture was approximately **6.1× slower on CPU** (691.70 ms/sample vs. 113.42 ms/sample). Model B is therefore superior in accuracy on complex nuances, but introduces substantial latency overhead. In latency-critical SLA environments, Model A remains an attractive high-throughput alternative.

---

## 7. Decoupled Production Inference Workflow (Phase 5)

In real-world deployment, downstream client applications must never be tightly coupled to specific model versions, file system paths, or model frameworks. Phase 5 validates this architectural decoupling:

- **Input:** `data/new_data.csv` (strictly unseen, unlabeled production examples)
- **Champion Model URI:** `models:/HF_Sentiment_Champion_Challenger@champion`
- **Prediction Script:** `predict_from_mlflow.py`
- **Output:** `outputs/new_data_predictions.csv`

### Architectural Decoupling Principles

1. **Dynamic Alias Resolution:**  
   The prediction script connects to the MLflow Model Registry and resolves `@champion`. It dynamically discovers the currently promoted version (`Version 2`) and its associated metadata without hard-coding version numbers (`/2`) or run IDs.
2. **Zero Direct Hugging Face Inference:**  
   The script does not invoke `transformers.pipeline`, `AutoTokenizer`, or `AutoModel`. Inference is performed entirely through the MLflow PyFunc interface (`champion_model.predict(...)`), proving the artifact is self-contained.
3. **Data Integrity & Metric Protection:**  
   The brand-new examples in `data/new_data.csv` have zero overlap with `data/eval_inputs.csv` and do not contain ground-truth labels. No metrics (Accuracy or F1) are calculated on new data; only sentiment classifications and confidence scores are emitted.

```
data/new_data.csv (unseen texts)
           │
           ▼
predict_from_mlflow.py
   ├── 1. Connect to MLflow: http://127.0.0.1:5000
   ├── 2. Resolve URI: models:/HF_Sentiment_Champion_Challenger@champion
   ├── 3. Discover: Version 2 (siebert/sentiment-roberta-large-english)
   └── 4. Load PyFunc & execute prediction (No direct HF imports)
           │
           ▼
outputs/new_data_predictions.csv (id, text, prediction, confidence)
```

*(Note: This demonstration validates MLOps decoupling and reproducibility; production deployment would require an inference server such as Triton or MLflow Model Serving with automated API rate limits).*

---

## 8. Project Structure

```
-1MS25CI018--mlflow-champion-challenger/
├── data/
│   ├── eval_inputs.csv             # 100-sample balanced SST-2 evaluation benchmark
│   └── new_data.csv                # 5 unseen production inference inputs (unlabeled)
├── outputs/
│   ├── predictions_A.csv           # Model A evaluation predictions (100 samples)
│   ├── predictions_B.csv           # Model B evaluation predictions (100 samples)
│   ├── metrics_summary.json        # Comparative metrics summary JSON
│   ├── new_data_predictions.csv    # Champion inference outputs on unseen data (Phase 5)
│   └── plots/
│       └── model_comparison.png    # Accuracy & Macro-F1 comparison chart
├── evaluate.py                     # Evaluation and MLflow experiment tracking pipeline
├── register_champion.py            # Model registration & alias management script
├── predict_from_mlflow.py          # Decoupled prediction via MLflow Champion URI
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation & viva reference
└── .gitignore                      # Git exclusion rules
```

---

## 9. Current Implementation Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Repository inspection, directory scaffolding, `.gitignore`, `requirements.txt`, base `README.md` | **Completed** |
| **Phase 2** | Evaluation datasets (`eval_inputs.csv`, `new_data.csv`), baseline `evaluate.py` with offline dev mode, prediction CSVs, JSON summary, and comparison plot | **Completed** |
| **Phase 3** | MLflow experiment tracking (`A_sentiment_distilbert`, `B_sentiment_roberta`, `compare_sentiment`) with parameters, metrics, tags, and artifacts | **Completed** |
| **Validation Gate** | 100-sample SST-2 benchmark generation, final evaluation run, input sequence verification, and rule evaluation ($\Delta\text{F1} = +0.0199 \ge 0.01 \rightarrow$ Challenger wins) | **Completed** |
| **Phase 4** | Model registration (`HF_Sentiment_Champion_Challenger`), PyFunc packaging, `@champion` / `@challenger` aliasing, alias resolution, and smoke tests | **Completed** |
| **Phase 5** | Decoupled Champion inference (`predict_from_mlflow.py`) loading Champion via MLflow alias to predict unseen data (`new_data_predictions.csv`) | **Completed** |


---

## 10. Reproduction & Execution Guide

Follow these exact steps to reproduce the entire Champion–Challenger pipeline from scratch.

### Step 1: Environment Setup & Dependencies

```bash
# 1. Clone repository and navigate to root directory
cd -1MS25CI018--mlflow-champion-challenger

# 2. Create and activate a Python virtual environment (Python 3.10+ recommended)
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

### Step 2: Start MLflow Tracking Server

The MLflow tracking server uses a local SQLite backend database and file-based artifact storage:

```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Verify that the MLflow UI is accessible in your browser at `http://127.0.0.1:5000`.

### Step 3: Run Model Evaluation & MLflow Experiment Tracking

To execute the full 100-sample SST-2 benchmark evaluation and log all runs (`A_sentiment_distilbert`, `B_sentiment_roberta`, and `compare_sentiment`) to MLflow:

```bash
python evaluate.py --mode full
```

*(Optional: Run `python evaluate.py --mode dev` to run a fast 10-sample verification smoke-test).*

### Step 4: Register Models & Assign Governance Aliases

To register both candidate models under `HF_Sentiment_Champion_Challenger` and programmatically assign the `@champion` alias to the winning model and `@challenger` to the runner-up:

```bash
python register_champion.py
```

### Step 5: Perform Decoupled Production Prediction

To predict sentiment for unseen production texts in `data/new_data.csv` using the dynamically resolved Champion model from MLflow:

```bash
python predict_from_mlflow.py
```

Predictions will be saved to `outputs/new_data_predictions.csv`.

---

## 11. Viva Voce & Technical Defense Guide

This section prepares students to defend the design, implementation, and empirical findings during academic evaluation or viva voce:

### Q1: Why evaluate Model A (DistilBERT) and Model B (RoBERTa-large)?
**Answer:** They represent a classic production trade-off between **operational efficiency** and **representational capacity**. Model A (DistilBERT, 66M parameters) represents an ultra-fast, lightweight baseline suitable for high-throughput edge or CPU inference. Model B (RoBERTa-large, ~355M parameters) represents a high-capacity deep transformer capable of capturing subtle linguistic nuances and negations at the cost of significantly higher latency and computational resource usage.

### Q2: Why must the Champion–Challenger decision rule be predefined?
**Answer:** In professional MLOps and clinical/financial ML systems, pre-registering decision criteria and minimum required margins ($\Delta \ge 0.01$) prevents confirmation bias, p-hacking, and post-hoc threshold manipulation. A new challenger must empirically justify the risk and computational switching cost of replacing the stable incumbent champion.

### Q3: Why is Macro-F1 the primary evaluation metric rather than simple Accuracy?
**Answer:** While our benchmark is balanced (50 positive / 50 negative), Macro-F1 evaluates precision and recall unweighted across both classes. In real-world sentiment analysis where class distributions drift or become imbalanced, accuracy can be deceptively high if a model collapses to the majority class. Macro-F1 penalizes asymmetric errors equally.

### Q4: Why package both models under ONE registered model name using aliases?
**Answer:** Production architectures require a single, stable logical entity (e.g., `HF_Sentiment_Champion_Challenger`) for a business capability. Downstream consumers should not change code or endpoints when models are retrained or swapped. By using MLflow Aliases (`@champion` and `@challenger`), governance teams can seamlessly promote a new model version to `@champion` without requiring downstream redeployment or config modifications.

### Q5: Why is direct Hugging Face inference prohibited in `predict_from_mlflow.py`?
**Answer:** Direct coupling to `transformers.pipeline` or local weights bypasses the MLOps governance layer. By loading models via `mlflow.pyfunc.load_model("models:/<Model>@<Alias>")`, the inference script demonstrates complete framework independence, verifies that the artifact logged in MLflow is self-contained and reproducible, and ensures that lineage and model versioning are strictly honored.

### Q6: What was the primary finding and engineering trade-off observed?
**Answer:** Model B achieved higher classification performance (+1.99% Macro-F1 and +2.0% Accuracy), crossing the $+0.01$ improvement margin to legitimately win the Champion title. However, its 24-layer architecture was **6.1× slower** on CPU (691.70 ms vs. 113.42 ms per sample). In production, an engineering team would assess whether the 2% accuracy gain justifies the 6x increase in compute costs and latency SLAs, potentially opting for GPU acceleration or model quantization (e.g., ONNX / TensorRT).

---

## 12. AI Assistant Disclosure

In adherence to academic integrity and project transparency guidelines:
- An AI coding assistant (Google Antigravity) was utilized during the development of this project for repository scaffolding, debugging platform-specific console encoding on Windows, implementing idiomatic MLflow tracking and PyFunc wrappers, and structuring documentation.
- All evaluation benchmarks (`eval_inputs.csv`), model inferences, empirical metrics, MLflow runs, model registrations, and prediction outputs were executed directly on local compute with verified models and deterministic seeds; **no metrics or predictions were synthetically fabricated or hardcoded**.
