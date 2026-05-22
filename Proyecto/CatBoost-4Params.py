# =========================================================
# LIBRERÍAS
# =========================================================
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error
)

from catboost import CatBoostClassifier

sns.set(style="whitegrid")

# =========================================================
# RUTAS
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "icfes_transformado.csv")

# =========================================================
# CARGAR DATA
# =========================================================
df = pd.read_csv(DATA_PATH)

# =========================================================
# TARGET
# =========================================================
def categorizar(x):
    x = float(x)
    if x <= 20:
        return 0
    elif x <= 40:
        return 1
    elif x <= 60:
        return 2
    elif x <= 80:
        return 3
    else:
        return 4

df["target"] = df["PERCENTIL_GLOBAL"].apply(categorizar)

# =========================================================
# FEATURES
# =========================================================
features = [col for col in df.columns if (
    col.startswith("FAMI_") or
    col.startswith("COLE_") or
    col.startswith("ESTU_")
)]

features = [col for col in features if not (
    col.startswith("PUNT_") or
    col.startswith("PERCENTIL_")
)]

X = df[features].copy()
y = df["target"].copy()

# =========================================================
# IDENTIFICAR TIPOS
# =========================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

# =========================================================
# LIMPIEZA
# =========================================================
for col in cat_cols:
    X[col] = X[col].astype(str)
    X[col] = X[col].replace("nan", "missing")

for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# =========================================================
# SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 🔥 4 CONJUNTOS DE PARÁMETROS (TUNING)
# =========================================================
param_grid = [

    # 🔹 CONJUNTO 1 (rápido / simple)
    {
        "iterations": [150],
        "learning_rate": [0.1],
        "depth": [4],
        "l2_leaf_reg": [1],
        "bootstrap_type": ['Bernoulli'],
        "subsample": [0.7]
    },

    # 🔹 CONJUNTO 2 (más profundo)
    {
        "iterations": [500],
        "learning_rate": [0.03],
        "depth": [8],
        "l2_leaf_reg": [5],
        "bootstrap_type": ['Bernoulli'],
        "subsample": [0.9]
    },

    # 🔹 CONJUNTO 3 (más regularizado)
    {
        "iterations": [600],
        "learning_rate": [0.01],
        "depth": [10],
        "l2_leaf_reg": [9],
        "bootstrap_type": ['Bernoulli'],
        "subsample": [0.8]
    },

    # 🔹 CONJUNTO 4 (TU CONFIGURACIÓN ORIGINAL)
    {
        "iterations": [300],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [3],
        "bootstrap_type": ['Bernoulli'],
        "subsample": [0.8]
    }
]

# =========================================================
# MODELO BASE
# =========================================================
cat_base = CatBoostClassifier(
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)

# =========================================================
# CROSS VALIDATION (10 FOLDS)
# =========================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=cat_base,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

# =========================================================
# ENTRENAMIENTO
# =========================================================
start = time.time()

grid.fit(
    X_train,
    y_train,
    cat_features=cat_indices
)

end = time.time()

# Mejor modelo
cat = grid.best_estimator_

print("\nMEJORES PARÁMETROS:\n")
print(grid.best_params_)

# =========================================================
# PREDICCIÓN
# =========================================================
y_pred = cat.predict(X_test)
y_proba = cat.predict_proba(X_test)

# =========================================================
# MÉTRICAS
# =========================================================
results = {
    "Modelo": "CatBoost",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred, average='weighted'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr'),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Tiempo": end - start
}

print("\nRESULTADOS:\n")
print(results)

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("CatBoost")
plt.savefig(os.path.join(BASE_DIR, "CatBoost", "catboost_confusion.png"))
plt.show()

# =========================================================
# GUARDAR RESULTADOS
# =========================================================
os.makedirs(os.path.join(BASE_DIR, "CatBoost"), exist_ok=True)

pd.DataFrame([results]).to_csv(
    os.path.join(BASE_DIR, "CatBoost", "catboost_metrics.csv"),
    index=False
)

pd.DataFrame(cm).to_csv(
    os.path.join(BASE_DIR, "CatBoost", "catboost_confusion.csv"),
    index=False
)
