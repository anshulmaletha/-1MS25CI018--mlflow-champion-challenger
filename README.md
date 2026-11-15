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

## 6. Project Structure

```
-1MS25CI018--mlflow-champion-challenger/
├── data/
│   ├── eval_inputs.csv             # 100-sample balanced SST-2 evaluation benchmark
│   └── new_data.csv                # 5 unseen production inference inputs (unlabeled)
├── outputs/
│   ├── predictions_A.csv           # Model A evaluation predictions (100 samples)
│   ├── predictions_B.csv           # Model B evaluation predictions (100 samples)
│   ├── metrics_summary.json        # Comparative metrics summary JSON
│   ├── new_data_predictions.csv    # Champion inference outputs
│   └── plots/
│       └── model_comparison.png    # Accuracy & Macro-F1 comparison chart
├── evaluate.py                     # Evaluation and MLflow experiment tracking pipeline
├── register_champion.py            # Model registration & alias management script
├── predict_from_mlflow.py          # Production prediction via MLflow URI
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation & viva reference
└── .gitignore                      # Git exclusion rules
```

---

## 7. Current Implementation Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Repository inspection, directory scaffolding, `.gitignore`, `requirements.txt`, base `README.md` | **Completed** |
| **Phase 2** | Evaluation datasets (`eval_inputs.csv`, `new_data.csv`), baseline `evaluate.py` with offline dev mode, prediction CSVs, JSON summary, and comparison plot | **Completed** |
| **Phase 3** | MLflow experiment tracking (`A_sentiment_distilbert`, `B_sentiment_roberta`, `compare_sentiment`) with parameters, metrics, tags, and artifacts | **Completed** |
| **Validation Gate** | 100-sample SST-2 benchmark generation, final evaluation run, input sequence verification, and rule evaluation ($\Delta\text{F1} = +0.0199 \ge 0.01 \rightarrow$ Challenger wins) | **Completed** |
| **Phase 4** | Model registration and Champion–Challenger aliasing | Pending |
| **Phase 5** | Decoupled Champion inference from MLflow on new data | Pending |
