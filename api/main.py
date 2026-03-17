import json
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

app = FastAPI(title="Sentiment140 API")

model = joblib.load("models/best_model.pkl")

class PredictRequest(BaseModel):
    texts: Union[str, List[str]]

@app.get("/model_info")
def model_info():
    with open("models/best_model_card.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/predict")
def predict(req: PredictRequest):
    texts = req.texts if isinstance(req.texts, list) else [req.texts]
    preds = model.predict(texts).tolist()
    return {"predictions": preds}

@app.get("/ablation_summary")
def ablation_summary():
    df = pd.read_csv("models/ablation_results.csv")
    return {
        "table": df.to_dict(orient="records"),
        "plot_file": "models/ablation_plot.png",
        "conclusions": "Se compararon baseline, TF-IDF unigram, TF-IDF bigram y Logistic Regression. El mejor modelo fue tfidf_bi_logreg según accuracy de validación.",
        "member": "Daniel"
    }

@app.get("/comparison")
def comparison():
    with open("models/comparison.json", "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/work_distribution")
def work_distribution():
    return {
        "distribution": [
            {
                "member": "Daniel",
                "tasks": [
                    "baseline",
                    "registro en MLflow",
                    "ablacion de codificacion",
                    "seleccion del mejor modelo",
                    "comparacion con Hugging Face",
                    "FastAPI"
                ]
            }
        ]
    }
