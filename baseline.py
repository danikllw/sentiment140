import re
import time
import random
import joblib
import mlflow
import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
test_df = pd.DataFrame(dataset["test"])

print("Columnas train:")
print(train_df.columns.tolist())
print(train_df.head())

TEXT_COL = "text"
LABEL_COL = "label"

train_df["text_clean"] = train_df[TEXT_COL].apply(basic_clean)
test_df["text_clean"] = test_df[TEXT_COL].apply(basic_clean)

train_df.to_csv("data/raw/train.csv", index=False)
test_df.to_csv("data/raw/test.csv", index=False)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_df["text_clean"],
    train_df[LABEL_COL],
    test_size=0.2,
    random_state=SEED,
    stratify=train_df[LABEL_COL]
)

mlflow.set_experiment("sentiment140_lab")

baseline = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("clf", MultinomialNB())
])

start = time.time()
baseline.fit(X_train, y_train)
elapsed = time.time() - start

preds = baseline.predict(X_valid)
acc = accuracy_score(y_valid, preds)

with mlflow.start_run(run_name="baseline_bow_nb"):
    mlflow.set_tag("member", "Daniel")
    mlflow.log_param("preprocessing", "basic_clean")
    mlflow.log_param("encoding", "bow")
    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("train_time_sec", elapsed)

print("Baseline accuracy:", acc)

joblib.dump(baseline, "models/baseline_model.pkl")
print("Modelo baseline guardado en models/baseline_model.pkl")
