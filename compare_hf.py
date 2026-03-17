import re
import time
import json
import joblib
import mlflow
import pandas as pd

from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import accuracy_score

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cargando dataset...")
dataset = load_dataset("adilbekovich/Sentiment140Twitter")
test_df = pd.DataFrame(dataset["test"])

TEXT_COL = "text"
LABEL_COL = "label"

test_df["text_clean"] = test_df[TEXT_COL].apply(basic_clean)

print("Cargando mejor modelo local...")
best_model = joblib.load("models/best_model.pkl")

subset_size = 2000
subset_df = test_df.sample(n=subset_size, random_state=42).copy()

print("Evaluando modelo local...")
start = time.time()
preds_local = best_model.predict(subset_df["text_clean"])
local_time = time.time() - start
local_acc = accuracy_score(subset_df[LABEL_COL], preds_local)

print("Cargando pipeline de Hugging Face...")
hf = pipeline("sentiment-analysis")

texts = subset_df[TEXT_COL].astype(str).tolist()
y_true = subset_df[LABEL_COL].tolist()

print("Evaluando Hugging Face...")
start = time.time()
hf_out = hf(texts, truncation=True, batch_size=32)
hf_time = time.time() - start

hf_preds = []
for item in hf_out:
    lbl = item["label"].upper()
    if "POSITIVE" in lbl:
        hf_preds.append(1)
    else:
        hf_preds.append(0)

hf_acc = accuracy_score(y_true, hf_preds)

comparison = {
    "subset_size": subset_size,
    "local_model_name": "tfidf_bi_logreg",
    "local_model_accuracy": float(local_acc),
    "local_model_time_sec": float(local_time),
    "hf_model_accuracy": float(hf_acc),
    "hf_model_time_sec": float(hf_time),
    "member": "Daniel"
}

with open("models/comparison.json", "w", encoding="utf-8") as f:
    json.dump(comparison, f, indent=2, ensure_ascii=False)

mlflow.set_experiment("sentiment140_lab")
with mlflow.start_run(run_name="comparison_hf"):
    mlflow.set_tag("member", "Daniel")
    mlflow.log_param("subset_size", subset_size)
    mlflow.log_param("local_model_name", "tfidf_bi_logreg")
    mlflow.log_metric("local_model_accuracy", local_acc)
    mlflow.log_metric("local_model_time_sec", local_time)
    mlflow.log_metric("hf_model_accuracy", hf_acc)
    mlflow.log_metric("hf_model_time_sec", hf_time)

print("\nComparacion:")
print(comparison)
print("\nGuardado en models/comparison.json")
