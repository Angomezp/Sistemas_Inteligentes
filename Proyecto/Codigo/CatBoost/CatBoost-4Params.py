import os
import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error
)

from catboost import CatBoostClassifier

sns.set(style="whitegrid")

# =========================================================
# RUTAS
# =========================================================
DATA_DIR = os.path.join("CSV", "icfes_transformado.csv")
OUTPUT_DIR = os.path.join("Resultados", "CatBoost")

# =========================================================
# CARGAR DATA
# =========================================================
df = pd.read_csv(DATA_DIR)


# =========================================================
# CONVERSIÓN DE TIPOS
# =========================================================
df = df.apply(
    lambda col: pd.to_numeric(
        col,
        errors="ignore"
    )
)

# =========================================================
# TARGET
# =========================================================
df["target"] = pd.qcut(
    df["PUNT_GLOBAL"],
    q=5,
    labels=False
)

# =========================================================
# FEATURES
# =========================================================
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

y = df["target"].copy()

# =========================================================
# TIPOS
# =========================================================
cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

# =========================================================
# LIMPIEZA
# =========================================================
for col in cat_cols:

    X[col] = X[col].astype(str)

    X[col] = X[col].replace(
        "nan",
        "missing"
    )

for col in num_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )

# =========================================================
# ÍNDICES DE VARIABLES CATEGÓRICAS
# =========================================================
cat_indices = [
    X.columns.get_loc(col)
    for col in cat_cols
]

# =========================================================
# SPLIT SOLO PARA EVALUACIÓN
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
# 4 SETS DE HIPERPARÁMETROS
# =========================================================
param_grid = [

    # =====================================================
    # SET 1
    # =====================================================
    {
        "iterations": [100],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [8]
    },
    # =====================================================
    # SET 2
    # =====================================================
    {
        "iterations": [100],
        "learning_rate": [0.1],
        "depth": [8],
        "l2_leaf_reg": [3]
    },

    # =====================================================
    # SET 3
    # =====================================================
    {
        "iterations": [150],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [8]
    },

    # =====================================================
    # SET 4
    # =====================================================
    {
        "iterations": [150],
        "learning_rate": [0.1],
        "depth": [8],
        "l2_leaf_reg": [3]
    }
]

# =========================================================
# TEN FOLD CROSS VALIDATION
# =========================================================
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# =========================================================
# GRID SEARCH
# =========================================================
grid_search = GridSearchCV(
    estimator=cat_base,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=2,
    verbose=2
)

# =========================================================
# ENTRENAMIENTO
# =========================================================
start = time.time()
grid_search.fit(X_train, y_train, cat_features=cat_indices)
end = time.time()

# =========================================================
# MEJORES PARÁMETROS
# =========================================================
print("\nMEJORES PARÁMETROS:\n")

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# =========================================================
# PREDICCIONES
# =========================================================
y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)

# =========================================================
# MÉTRICAS
# =========================================================
results = {
    "Modelo": "CatBoost",

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

# =========================================================
# MOSTRAR RESULTADOS
# =========================================================
print("\nRESULTADOS:\n")

print(results)

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = confusion_matrix(y_test, y_pred)

# =========================================================
# CREAR CARPETA
# =========================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# HEATMAP
# =========================================================
plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("CatBoost")

plt.xlabel("Predicción")

plt.ylabel("Real")

plt.savefig(os.path.join(OUTPUT_DIR,"catboost_confusion.png"), bbox_inches='tight', dpi=150)
plt.close()

# =========================================================
# EXPORTAR MÉTRICAS
# =========================================================
pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "catboost_metrics.csv"), index=False)

# =========================================================
# EXPORTAR MATRIZ
# =========================================================
pd.DataFrame(cm).to_csv( os.path.join(OUTPUT_DIR, "catboost_confusion.csv"), index=False)

# =========================================================
# EXPORTAR MEJORES HIPERPARÁMETROS
# =========================================================
pd.DataFrame([grid_search.best_params_]).to_csv( os.path.join( OUTPUT_DIR, "catboost_best_params.csv"), index=False)

# =========================================================
# REENTRENAR CON TODO EL DATASET
# =========================================================
final_model = best_model.fit(X, y, cat_features=cat_indices)

# =========================================================
# FINAL
# =========================================================

joblib.dump(final_model, os.path.join(OUTPUT_DIR, "catboost_final_model.pkl"))

print("\nModelo CatBoost entrenado y evaluado con éxito. Resultados y archivos exportados en: ", OUTPUT_DIR)