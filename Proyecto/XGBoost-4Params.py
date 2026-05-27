# =========================================================
# LIBRERÍAS
# =========================================================
import os
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

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error
)

sns.set(style="whitegrid")

# =========================================================
# CARGAR DATA
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "icfes_transformado.csv")

df = pd.read_csv(DATA_DIR)

# =========================================================
# CONVERSIÓN DE TIPOS
# =========================================================
df = df.apply(lambda col: pd.to_numeric(col, errors="ignore"))

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
y = df["target"]

# =========================================================
# COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# =========================================================
num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

# =========================================================
# LIMPIEZA
# =========================================================
for col in num_cols:

    X.loc[:, col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X.loc[:, col] = X[col].fillna(
        X[col].median()
    )

for col in cat_cols:

    X.loc[:, col] = X[col].astype(str)

    X.loc[:, col] = X[col].replace(
        "nan",
        "missing"
    )

# =========================================================
# PREPROCESAMIENTO
# =========================================================
preprocessor = ColumnTransformer([

    (
        'num',
        StandardScaler(),
        num_cols
    ),

    (
        'cat',
        OneHotEncoder(handle_unknown='ignore'),
        cat_cols
    )

])

# =========================================================
# SPLIT PARA EVALUACIÓN
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# =========================================================
# PIPELINE XGBOOST
# =========================================================
xgb_pipe = Pipeline([

    ('prep', preprocessor),

    ('model', XGBClassifier(

        objective='multi:softprob',

        num_class=5,

        eval_metric='mlogloss',

        random_state=42,

        n_jobs=-1

    ))

])

# =========================================================
# 4 GRUPOS DE HIPERPARÁMETROS
# =========================================================
param_grid = {

    # Grupo 1 -> Complejidad
    "model__max_depth": [3, 5, 7, 10],

    # Grupo 2 -> Aprendizaje
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],

    # Grupo 3 -> Cantidad de árboles
    "model__n_estimators": [100, 200, 300],

    # Grupo 4 -> Generalización
    "model__subsample": [0.6, 0.8, 1.0]
}

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

    estimator=xgb_pipe,

    param_grid=param_grid,

    cv=cv,

    scoring='f1_weighted',

    n_jobs=-1,

    verbose=2
)

# =========================================================
# ENTRENAMIENTO
# =========================================================
start = time.time()

grid_search.fit(X_train, y_train)

end = time.time()

# =========================================================
# MEJORES PARÁMETROS
# =========================================================
print("\nMEJORES PARÁMETROS:\n")

print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# =========================================================
# REENTRENAR CON TODO EL DATASET
# =========================================================
final_model = best_model.fit(X, y)

# =========================================================
# PREDICCIONES
# =========================================================
y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)

# =========================================================
# MÉTRICAS
# =========================================================
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

# =========================================================
# MOSTRAR RESULTADOS
# =========================================================
print("\nRESULTADOS:\n")

print(results)

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = confusion_matrix(
    y_test,
    y_pred
)

# =========================================================
# CREAR CARPETA
# =========================================================
os.makedirs(

    os.path.join(
        BASE_DIR,
        "XGBOOST"
    ),

    exist_ok=True
)

# =========================================================
# HEATMAP
# =========================================================
plt.figure(figsize=(8,6))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"
)

plt.title("XGBoost")

plt.xlabel("Predicción")

plt.ylabel("Real")

plt.savefig(

    os.path.join(

        BASE_DIR,

        "XGBOOST",

        "xgboost_confusion.png"
    )
)

plt.show()

# =========================================================
# EXPORTAR MÉTRICAS
# =========================================================
pd.DataFrame([results]).to_csv(

    os.path.join(

        BASE_DIR,

        "XGBOOST",

        "xgboost_metrics.csv"
    ),

    index=False
)

# =========================================================
# EXPORTAR MATRIZ
# =========================================================
pd.DataFrame(cm).to_csv(

    os.path.join(

        BASE_DIR,

        "XGBOOST",

        "xgboost_confusion.csv"
    ),

    index=False
)

# =========================================================
# EXPORTAR MEJORES HIPERPARÁMETROS
# =========================================================
pd.DataFrame([grid_search.best_params_]).to_csv(

    os.path.join(

        BASE_DIR,

        "XGBOOST",

        "xgboost_best_params.csv"
    ),

    index=False
)

# =========================================================
# FINAL
# =========================================================
print("\nMODELO FINAL ENTRENADO CON TODO EL DATASET")
