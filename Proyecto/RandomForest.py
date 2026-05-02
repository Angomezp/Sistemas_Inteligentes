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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error
)

sns.set(style="whitegrid")

# =========================================================
# CARGAR DATA (SIN dtype=str)
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "icfes_transformado.csv")

df = pd.read_csv(DATA_DIR)

# =========================================================
# CONVERSIÓN AUTOMÁTICA DE TIPOS
# =========================================================
df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

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
y = df["target"]

# =========================================================
# TIPOS CORRECTOS
# =========================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("Numéricas:", list(num_cols))
print("Categóricas:", list(cat_cols))

# =========================================================
# LIMPIEZA NaN
# =========================================================
# Numéricas
for col in num_cols:
    X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")
    X.loc[:, col] = X[col].fillna(X[col].median())

# Categóricas
for col in cat_cols:
    X.loc[:, col] = X[col].astype(str)
    X.loc[:, col] = X[col].replace("nan", "missing")

# =========================================================
# PREPROCESAMIENTO
# =========================================================
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# =========================================================
# SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# MODELO RANDOM FOREST
# =========================================================
rf = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_features='sqrt',
        max_depth=15,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

# =========================================================
# EVALUACIÓN
# =========================================================
def evaluate(name, model):

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return {
        "Modelo": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred, average='weighted'),
        "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr'),
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Tiempo": end-start,
        "Confusion": confusion_matrix(y_test, y_pred)
    }

# =========================================================
# ENTRENAR
# =========================================================
res_rf = evaluate("Random Forest", rf)

print("\nRESULTADOS:\n")
print(pd.DataFrame([res_rf]))

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = res_rf["Confusion"]

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest")
plt.savefig(os.path.join(BASE_DIR, "rf_confusion.png"))
plt.show()