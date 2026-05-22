# =========================================================
# LIBRERÍAS
# =========================================================
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
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
# TIPOS
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
# SPLIT (solo evaluación)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
# 🔥 DISTRIBUCIÓN DE HIPERPARÁMETROS
# (incluye tu configuración dentro del rango)
# =========================================================
param_dist = {
    "iterations": [150, 300, 500, 700],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "depth": [4, 6, 8, 10],
    "l2_leaf_reg": [1, 3, 5, 9],
    "bootstrap_type": ['Bernoulli'],
    "subsample": [0.7, 0.8, 0.9]
}

# =========================================================
# CROSS VALIDATION (3 folds)
# =========================================================
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=cat_base,
    param_distributions=param_dist,
    n_iter=6,  # puedes subir a 10-15 si quieres mejor resultado
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# =========================================================
# ENTRENAMIENTO (búsqueda)
# =========================================================
start = time.time()

random_search.fit(
    X_train,
    y_train,
    cat_features=cat_indices
)

end = time.time()

print("\nMEJORES PARÁMETROS:\n")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# =========================================================
# 🔥 REENTRENAR CON TODO EL DATASET
# =========================================================
final_model = best_model.fit(
    X,
    y,
    cat_features=cat_indices
)

# =========================================================
# EVALUACIÓN (referencia)
# =========================================================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

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

os.makedirs(os.path.join(BASE_DIR, "CatBoost"), exist_ok=True)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("CatBoost")
plt.savefig(os.path.join(BASE_DIR, "CatBoost", "catboost_confusion.png"))
plt.show()

pd.DataFrame([results]).to_csv(
    os.path.join(BASE_DIR, "CatBoost", "catboost_metrics.csv"),
    index=False
)

pd.DataFrame(cm).to_csv(
    os.path.join(BASE_DIR, "CatBoost", "catboost_confusion.csv"),
    index=False
)
