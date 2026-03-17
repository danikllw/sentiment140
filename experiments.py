import re
import time
import json
import random
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cargando dataset...")
dataset = load_dataset("adilbekovich/Sentiment140Twitter")
train_df = pd.DataFrame(dataset["train"])

TEXT_COL = "text"
LABEL_COL = "label"

train_df["text_clean"] = train_df[TEXT_COL].apply(basic_clean)

sample_df = train_df.sample(n=300000, random_state=SEED)

X_train, X_valid, y_train, y_valid = train_test_split(
    sample_df["text_clean"],
    sample_df[LABEL_COL],
    test_size=0.2,
    random_state=SEED,
    stratify=sample_df[LABEL_COL]
)

mlflow.set_experiment("sentiment140_lab")

experiments = {
    "baseline_bow_nb": Pipeline([
        ("vectorizer", CountVectorizer(max_features=30000)),
        ("clf", MultinomialNB())
    ]),
    "tfidf_uni_nb": Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 1), max_features=30000)),
        ("clf", MultinomialNB())
    ]),
    "tfidf_bi_nb": Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
        ("clf", MultinomialNB())
    ]),
    "tfidf_bi_logreg": Pipeline([
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
        ("clf", LogisticRegression(max_iter=300, random_state=SEED))
    ])
}

results = []

for name, model in experiments.items():
    print(f"Ejecutando {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_valid)
    acc = accuracy_score(y_valid, preds)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("member", "Daniel")
        mlflow.log_param("preprocessing", "basic_clean")
        mlflow.log_param("experiment_name", name)
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("train_time_sec", train_time)

    results.append({
        "experiment": name,
        "val_accuracy": acc,
        "train_time_sec": train_time
    })

results_df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False)
results_df.to_csv("models/ablation_results.csv", index=False)

best_name = results_df.iloc[0]["experiment"]
best_model = experiments[best_name]

print(f"Reentrenando mejor modelo con muestra: {best_name}")
best_model.fit(sample_df["text_clean"], sample_df[LABEL_COL])
joblib.dump(best_model, "models/best_model.pkl")

model_card = {
    "model_name": best_name,
    "metric": "accuracy",
    "best_val_accuracy": float(results_df.iloc[0]["val_accuracy"]),
    "preprocessing": "basic_clean",
    "member": "Daniel"
}

with open("models/best_model_card.json", "w", encoding="utf-8") as f:
    json.dump(model_card, f, indent=2, ensure_ascii=False)

plt.figure(figsize=(10, 5))
plt.bar(results_df["experiment"], results_df["val_accuracy"])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("models/ablation_plot.png")

print("\nResultados:")
print(results_df)
print(f"\nMejor modelo: {best_name}")
print("Guardados:")
print("- models/ablation_results.csv")
print("- models/best_model.pkl")
print("- models/best_model_card.json")
print("- models/ablation_plot.png")
