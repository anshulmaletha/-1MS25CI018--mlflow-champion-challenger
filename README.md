# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and Models

**Task:** Sentiment Analysis  
**Dataset:** Stanford Sentiment Treebank (SST-2) validation set  
**Evaluation subset:** 100 examples — 50 Positive and 50 Negative  
**Random seed:** 42

### Model A — Initial Champion
- **Hugging Face ID:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Role:** Initial Champion
- **Model type:** DistilBERT fine-tuned for SST-2 sentiment classification

### Model B — Challenger
- **Hugging Face ID:** `siebert/sentiment-roberta-large-english`
- **Role:** Challenger
- **Model type:** RoBERTa-large fine-tuned for English sentiment classification

Both models were evaluated on the **same 100 inputs in the same order** to ensure a fair comparison.

---

## 2. How to Run

Start the MLflow tracking server:

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then, from the project directory, run the evaluation:

```bash
python evaluate.py
```

Register the evaluated models and assign Champion/Challenger aliases:

```bash
python register_champion.py
```

Run inference on new, unseen data using the MLflow Champion alias:

```bash
python predict_from_mlflow.py
```

The final prediction output is saved to:

```text
outputs/new_data_predictions.csv
```

The Champion is loaded through the MLflow Model Registry using:

```text
models:/HF_Sentiment_Champion_Challenger@champion
```

---

## 3. Champion Rule

The Champion rule was defined **before the final benchmark run**.

- **Primary metric:** Macro-F1
- **Direction:** Higher is better
- **Minimum improvement margin:** 0.01
- **Tie-breaker:** Lower average inference latency
- **Decision rule:** The Challenger becomes Champion only if its Macro-F1 improves over the current Champion by at least 0.01. Otherwise, the current Champion remains Champion. If the primary metric is tied within the decision margin, lower latency is used as the tie-breaker.

Model A was initially designated as the Champion, while Model B was evaluated as the Challenger.

---

## 4. Results Table

| Metric | Model A — DistilBERT | Model B — RoBERTa | Difference (B − A) |
|---|---:|---:|---:|
| Accuracy | 0.9200 | 0.9400 | +0.0200 |
| Macro-F1 | 0.9200 | 0.9399 | +0.0199 |
| Avg. Latency (ms/sample) | 113.42 | 691.70 | +578.28 |
| Total Inference Time (s) | 11.3417 | 69.1695 | +57.8278 |

### Final Decision

**Champion:** Model B — `siebert/sentiment-roberta-large-english`

Model B achieved a Macro-F1 score of **0.9399**, compared with **0.9200** for Model A. The improvement of **0.0199** is greater than the predefined minimum improvement margin of **0.01**, so the Challenger satisfies the Champion rule and replaces Model A.

---

## 5. Decision

The final Champion is **Model B (`siebert/sentiment-roberta-large-english`)**. The decision was based primarily on Macro-F1, which was selected before running the final benchmark. Model B achieved a Macro-F1 of **0.9399**, while Model A (`distilbert-base-uncased-finetuned-sst-2-english`) achieved **0.9200**. This gives Model B an improvement of **0.0199**, exceeding the required improvement margin of **0.01**. Model B also achieved higher accuracy, with **0.9400** compared with **0.9200** for Model A.

However, the accuracy improvement comes with a significant latency trade-off. Model B required approximately **691.70 ms per sample**, whereas Model A required approximately **113.42 ms per sample**. Therefore, Model B is about 6 times slower on the CPU environment used for evaluation. Despite this performance cost, the improvement in Macro-F1 was large enough to satisfy the predefined Champion rule.

Based on the selected evaluation criteria, Model B is therefore promoted to **Champion**, while Model A becomes the **Challenger**. Both models were registered in MLflow, with the `champion` alias pointing to Model B and the `challenger` alias pointing to Model A. The Champion was subsequently loaded through the MLflow Model Registry and used to generate predictions on separate, previously unseen data.

---

## 6. Limitations and Declaration

### Limitations

The final comparison used a relatively small evaluation subset of **100 SST-2 validation examples**, so the measured performance may not represent performance on all possible sentiment inputs. The evaluation was performed in a CPU environment, making latency particularly sensitive to the hardware available. Model B provides better classification performance but has substantially higher inference latency and model size.

The evaluation data and new prediction data were kept separate. The 5 examples in `data/new_data.csv` were not used for calculating the benchmark metrics.

### AI Assistant Declaration

AI assistants were used during development for code guidance, debugging, documentation support, and project workflow assistance. The final implementation, MLflow tracking, model comparison, Champion–Challenger decision, registry configuration, and inference workflow were reviewed and verified by the student.

The project uses MLflow for experiment tracking and model management, and Hugging Face Transformers models for sentiment classification.
