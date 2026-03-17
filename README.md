# Sentiment140 Lab

## Objetivo
Implementar un sistema de análisis de sentimientos sobre Sentiment140, comparar configuraciones de codificación y modelos, registrar experimentos en MLflow y exponer el mejor modelo mediante FastAPI.

## Dataset
Se utilizó el dataset `adilbekovich/Sentiment140Twitter` desde Hugging Face.

## Métrica
Accuracy

## Baseline
- Preprocesamiento: limpieza mínima
- Codificación: Bag of Words
- Modelo: Multinomial Naive Bayes
- Accuracy baseline: 0.7712

## Experimentos realizados
- baseline_bow_nb
- tfidf_uni_nb
- tfidf_bi_nb
- tfidf_bi_logreg

## Mejor modelo
- Nombre: tfidf_bi_logreg
- Accuracy de validación: 0.8026166666666666

## Comparación con Hugging Face
- Modelo local accuracy: 0.787
- Hugging Face accuracy: 0.7015
- Tiempo modelo local: 0.0242 s
- Tiempo Hugging Face: 2.5617 s

## Estructura del proyecto
- `baseline.py`
- `experiments.py`
- `compare_hf.py`
- `models/`
- `api/`
- `notebooks/`

## MLflow
Ejecutar:
```bash
mlflow ui




