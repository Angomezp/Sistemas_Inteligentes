import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
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

from xgboost import XGBClassifier


sns.set(style="whitegrid")

# Paths
DATA_DIR = os.path.join("CSV", "icfes_transformado.csv")
OUTPUT_DIR = os.path.join("Resultados", "XGBoost")

df = pd.read_csv(DATA_DIR)


# Features de predicción (solo las que empiezan con FAMI_, COLE_ o ESTU_, excluyendo las que empiezan con PUNT_ o PERCENTIL_)
features = [
    col for col in df.columns if ( 
        col.startswith("FAMI_") or
        col.startswith("COLE_") or
        col.startswith("ESTU_")
    )
]

features = [
    col for col in features if not (
        col.startswith("PUNT_") or
        col.startswith("PERCENTIL_")
    )
]

X = df[features].copy()

y = df["target"]

# Columnas numéricas y categóricas

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

# Preprocesamiento para el metodo XGBoost (escala para numéricas, one-hot encoding para categóricas)
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
])

# split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Crear Pipeline
xgb_pipe = Pipeline([
    ("prep", preprocessor),
    (
        "model",
        XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
            max_depth=15,
            learning_rate=0.1,
            n_estimators=300,
            subsample=0.6
        ),
    ),
])

# Sets de hiperparámetros para probar (4 configuraciones)
param_grid = [

    # Set 1
    {
        "model__max_depth": [12],
        "model__learning_rate": [0.05],
        "model__n_estimators": [300],
        "model__subsample": [0.6]
    },

    # Set 2
    {
        "model__max_depth": [12],
        "model__learning_rate": [0.08],
        "model__n_estimators": [250],
        "model__subsample": [0.8]
    },

    # Set 3
    {
        "model__max_depth": [15],
        "model__learning_rate": [0.1],
        "model__n_estimators": [300],
        "model__subsample": [0.6]
    },

    # Set 4
    {
        "model__max_depth": [15],
        "model__learning_rate": [0.2],
        "model__n_estimators": [250],
        "model__subsample": [0.8]
    }
]

# Crear validación cruzada estratificada de 10 folds
cv = StratifiedKFold(

    n_splits=10,

    shuffle=True,

    random_state=42
)

# Grid Search (para probar las 4 configuraciones de hiperparámetros) con validación cruzada
grid_search = GridSearchCV(
    estimator=xgb_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=3,
    verbose=2,
)

# Entrenamiento y validacion con Grid Search (medición de tiempo incluida)
start = time.time()

grid_search.fit(X_train, y_train)
end = time.time()

# Mostrar mejores hiperparámetros encontrados
print("\nMEJORES PARÁMETROS:\n")

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predicciones con el mejor modelo encontrado
y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)

# Calcular métricas de evaluación
results = {

    "Modelo": "XGBoost",

    "Accuracy": accuracy_score(
        y_test,
        y_pred
    ),

    "F1": f1_score(
        y_test,
        y_pred,
        average='weighted'
    ),

    "ROC-AUC": roc_auc_score(
        y_test,
        y_proba,
        multi_class='ovr'
    ),

    "MAE": mean_absolute_error(
        y_test,
        y_pred
    ),

    "RMSE": np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    ),

    "Tiempo": end - start
}

# Mostrar resultados
print("\nRESULTADOS:\n")

print(results)

# Matriz de confusión
cm = confusion_matrix(
    y_test,
    y_pred
)

# Asegurar que la carpeta de resultados exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Alizar matriz de confusión con heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("XGBoost")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig(os.path.join(OUTPUT_DIR, "xgboost_confusion.png"))
plt.close()

# Exportar métricas a CSV
pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "xgboost_metrics.csv"), index=False)

pd.DataFrame(cm).to_csv(os.path.join(OUTPUT_DIR, "xgboost_confusion.csv"), index=False)

pd.DataFrame([grid_search.best_params_]).to_csv(os.path.join(OUTPUT_DIR, "xgboost_best_params.csv"), index=False)

# Reentrenar el modelo final con todos los datos (train + test) usando los mejores hiperparámetros encontrados
final_model = best_model.fit(X, y)

joblib.dump(best_model, os.path.join(OUTPUT_DIR, "xgboost_final_model.pkl"))

print("\nModelo XGBoost entrenado y evaluado con éxito. Resultados guardados en la carpeta " + OUTPUT_DIR + ".\n")
