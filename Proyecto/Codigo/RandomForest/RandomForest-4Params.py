import os
import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


sns.set(style="whitegrid")

# Paths
DATA_DIR = os.path.join("CSV", "icfes_transformado.csv")
OUTPUT_DIR = os.path.join("Resultados", "RandomForest")


df = pd.read_csv(DATA_DIR)


# Cargar features de predicción (solo las que empiezan con FAMI_, COLE_ o ESTU_, excluyendo las que empiezan con PUNT_ o PERCENTIL_)
features = [
    col for col in df.columns
    if (col.startswith("FAMI_") or col.startswith("COLE_") or col.startswith("ESTU_"))
]

features = [
    col for col in features 
    if not (
        col.startswith("PUNT_") or
        col.startswith("PERCENTIL_")
    )
]

X = df[features].copy()
y = df["target"]

# Seleccionar columnas numéricas y categóricas
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

# Limpieza de datos: numéricas con mediana, categóricas con "missing"
for col in num_cols:
    X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")
    X.loc[:, col] = X[col].fillna(X[col].median())

for col in cat_cols:
    X.loc[:, col] = X[col].astype(str)
    X.loc[:, col] = X[col].replace("nan", "missing")

# Preprocesamiento para el metodo Random Forest (escala para numéricas, one-hot encoding para categóricas) 
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
])

# Split solo de test y entrenamiento, Validacion se hace con GridSearchCV y StratifiedKFold
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline con preprocesamiento y modelo
rf_pipe = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=2),
    ),
])

# Set de hiperparámetros para probar (4 configuraciones)
param_grid = [

    # Set 1
    {
        "model__n_estimators": [200],
        "model__max_depth": [15],
        "model__max_features": ['sqrt'],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [2]
    },

    # Set 2
    {
        "model__n_estimators": [300],
        "model__max_depth": [20],
        "model__max_features": ['log2'],
        "model__min_samples_split": [5],
        "model__min_samples_leaf": [4]
    },

    # Set 3
    {
        "model__n_estimators": [400],
        "model__max_depth": [15],
        "model__max_features": ['log2'],
        "model__min_samples_split": [2],
        "model__min_samples_leaf": [2]
    },

    # Set 4
    {
        "model__n_estimators": [400],
        "model__max_depth": [20],
        "model__max_features": ['sqrt'],
        "model__min_samples_split": [5],
        "model__min_samples_leaf": [4]
    }
]

# Validación cruzada estratificada de 10 folds
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# Grid Search (para probar las 4 configuraciones de hiperparámetros) con validación cruzada
grid_search = GridSearchCV(
    estimator=rf_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=2,
    verbose=2
)

# Entrenar con diferentes hiperparámetros y medir tiempo
start = time.time()
grid_search.fit(X_train, y_train)
end = time.time()

# Imprimir mejores parámetros
print("\nMEJORES PARÁMETROS:\n")
print(grid_search.best_params_)
best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

# Calcular metricas
results = {
    "Modelo": "Random Forest",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred, average="weighted"),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class="ovr"),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Tiempo": end - start,
}

# Mostrar resultados en consola
print("\nRESULTADOS:\n")
print(results)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Crear Directorio de Resultados si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Heatmap de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig(os.path.join(OUTPUT_DIR, "rf_confusion_matrix.png"))
plt.close()

# Exportar métricas a CSV
pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "rf_metrics.csv"), index=False)

# Exportar matriz de confusión
pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, "rf_confusion.csv"), index=False)

# Exportar mejores parámetros
pd.DataFrame([grid_search.best_params_]).to_csv(os.path.join(OUTPUT_DIR, "rf_best_params.csv"), index=False)


# Exportar modelo final entrenado con todos los datos
final_model = best_model.fit(X, y)

joblib.dump(final_model, os.path.join(OUTPUT_DIR, "random_forest.pkl"))


print("\nModelo final entrenado y guardado en:", os.path.join(OUTPUT_DIR, "random_forest.pkl"))