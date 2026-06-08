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

# Paths
DATA_DIR = os.path.join("CSV", "icfes_transformado.csv")
OUTPUT_DIR = os.path.join("Resultados", "CatBoost")

# Cargar dataset
df = pd.read_csv(DATA_DIR)

# Seleccionar solo las columnas relevantes
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

# Variables predictoras y variable objetivo
X = df[features].copy()

y = df["target"].copy()

# Identificar columnas categóricas y numéricas
cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

# Índices de columnas categóricas para CatBoost
cat_indices = [
    X.columns.get_loc(col)
    for col in cat_cols
]

# Split en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo base
cat_base = CatBoostClassifier(

    loss_function='MultiClass',
    auto_class_weights='Balanced',
    random_state=42,
    verbose=0
)

# Grid de hiperparámetros
param_grid = [

    # Set 1
    {
        "iterations": [100],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [8]
    },
    # Set 2
    {
        "iterations": [100],
        "learning_rate": [0.1],
        "depth": [8],
        "l2_leaf_reg": [3]
    },

    # Set 3
    {
        "iterations": [150],
        "learning_rate": [0.05],
        "depth": [6],
        "l2_leaf_reg": [8]
    },

    # Set 4
    {
        "iterations": [150],
        "learning_rate": [0.1],
        "depth": [8],
        "l2_leaf_reg": [3]
    }
]

# Ten fold cross-validation
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# GridSearch para encontrar los mejores hiperparámetros con Ten Fold Cross Validation
grid_search = GridSearchCV(
    estimator=cat_base,
    param_grid=param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=2,
    verbose=2
)

# Entrenamiento y validación con GridSearchCV
start = time.time()
grid_search.fit(X_train, y_train, cat_features=cat_indices)
end = time.time()

# Exponer mejores parametros
print("\nMEJORES PARÁMETROS:\n")

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Predicciones
y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)

# Calcular métricas
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

# Resultados
print("\nRESULTADOS:\n")

print(results)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Asegurar que el directorio de salida existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualizar y guardar matriz de confusión
plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("CatBoost")

plt.xlabel("Predicción")

plt.ylabel("Real")

plt.savefig(os.path.join(OUTPUT_DIR,"catboost_confusion.png"), bbox_inches='tight', dpi=150)
plt.close()

# Exportar datos
pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "catboost_metrics.csv"), index=False)

pd.DataFrame(cm).to_csv( os.path.join(OUTPUT_DIR, "catboost_confusion.csv"), index=False)

pd.DataFrame([grid_search.best_params_]).to_csv( os.path.join( OUTPUT_DIR, "catboost_best_params.csv"), index=False)

# Reentrenar el modelo final con los mejores hiperparámetros en todo el conjunto de datos
final_model = best_model.fit(X, y, cat_features=cat_indices)

# Exportar el modelo final

joblib.dump(final_model, os.path.join(OUTPUT_DIR, "catboost_final_model.pkl"))

print("\nModelo CatBoost entrenado y evaluado con éxito. Resultados y archivos exportados en: ", OUTPUT_DIR)