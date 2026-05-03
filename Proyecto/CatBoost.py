# =========================================================
# LIBRERÍAS
# =========================================================
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
    if x <= 33:
        return 0
    elif x <= 66:
        return 1
    else:
        return 2

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
# IDENTIFICAR TIPOS CORRECTAMENTE
# =========================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

print("Categóricas:", list(cat_cols))
print("Numéricas:", list(num_cols))

# =========================================================
# LIMPIEZA DE NaN
# =========================================================

# Categóricas → string + valor "missing"
for col in cat_cols:
    X[col] = X[col].astype(str)
    X[col] = X[col].replace("nan", "missing")

# Numéricas → convertir + imputar
for col in num_cols:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(X[col].median())

# Índices de categóricas
cat_indices = [X.columns.get_loc(col) for col in cat_cols]

# =========================================================
# SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# MODELO CATBOOST
# =========================================================
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',   
    subsample=0.8,                
    loss_function='MultiClass',
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)

# =========================================================
# ENTRENAMIENTO
# =========================================================
start = time.time()

cat.fit(
    X_train,
    y_train,
    cat_features=cat_indices
)

end = time.time()

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
pd.DataFrame([results]).to_csv(
    os.path.join(BASE_DIR,"CatBoost" ,"catboost_metrics.csv"),
    index=False
)

pd.DataFrame(cm).to_csv(
    os.path.join(BASE_DIR, "CatBoost", "catboost_confusion.csv"),
    index=False
)