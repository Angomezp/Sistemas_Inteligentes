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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error
)

sns.set(style="whitegrid")

# =========================================================
# CARGAR DATA
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "icfes_transformado.csv")

df = pd.read_csv(DATA_DIR)

# =========================================================
# TIPOS
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
# TIPOS
# =========================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

# =========================================================
# LIMPIEZA
# =========================================================
for col in num_cols:
    X.loc[:, col] = pd.to_numeric(X[col], errors="coerce")
    X.loc[:, col] = X[col].fillna(X[col].median())

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
# SPLIT (solo evaluación)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# PIPELINE
# =========================================================
knn_pipe = Pipeline([
    ('prep', preprocessor),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

# =========================================================
# 🔥 HIPERPARÁMETROS
# =========================================================
param_dist = {
    "model__n_neighbors": [3, 5, 7, 9, 11, 15],
    "model__weights": ['uniform', 'distance'],
    "model__metric": ['euclidean', 'minkowski'],
    "model__p": [1, 2]
}

# =========================================================
# CROSS VALIDATION
# =========================================================
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    estimator=knn_pipe,
    param_distributions=param_dist,
    n_iter=6,  # puedes subir a 10 si quieres mejor búsqueda
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

random_search.fit(X_train, y_train)

end = time.time()

print("\nMEJORES PARÁMETROS:\n")
print(random_search.best_params_)

best_model = random_search.best_estimator_

# =========================================================
# 🔥 REENTRENAR CON TODO EL DATASET
# =========================================================
final_model = best_model.fit(X, y)

# =========================================================
# EVALUACIÓN (solo referencia)
# =========================================================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

results = {
    "Modelo": "KNN",
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred, average='weighted'),
    "ROC-AUC": roc_auc_score(y_test, y_proba, multi_class='ovr'),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "Tiempo": end-start
}

print("\nRESULTADOS:\n")
print(results)

# =========================================================
# MATRIZ DE CONFUSIÓN
# =========================================================
cm = confusion_matrix(y_test, y_pred)

os.makedirs(os.path.join(BASE_DIR, "KNN"), exist_ok=True)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("KNN")
plt.savefig(os.path.join(BASE_DIR, "KNN", "knn_confusion.png"))
plt.show()

pd.DataFrame([results]).to_csv(
    os.path.join(BASE_DIR, "KNN", "knn_metrics.csv"),
    index=False
)

pd.DataFrame(cm).to_csv(
    os.path.join(BASE_DIR, "KNN", "knn_confusion.csv"),
    index=False
)
