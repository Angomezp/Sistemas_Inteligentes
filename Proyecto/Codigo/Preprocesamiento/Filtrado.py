import os
import re
import numpy as np
import pandas as pd

# =========================================================
# RUTAS
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

INPUT_CSV = os.path.join(
    BASE_DIR,
    "CSV",
    "icfes_original.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "CSV",
    "icfes_limpio.csv"
)

# =========================================================
# COLUMNAS A UTILIZAR
# =========================================================
COLUMNAS = [
    "ESTU_DEPTO_RESIDE",
    "ESTU_MCPIO_RESIDE",
    "FAMI_ESTRATOVIVIENDA",
    "FAMI_PERSONASHOGAR",
    "FAMI_EDUCACIONPADRE",
    "FAMI_EDUCACIONMADRE",
    "FAMI_TIENEINTERNET",
    "FAMI_TIENECOMPUTADOR",
    "ESTU_DEDICACIONLECTURADIARIA",
    "ESTU_DEDICACIONINTERNET",
    "ESTU_HORASSEMANATRABAJA",
    "COLE_NATURALEZA",
    "COLE_BILINGUE",
    "COLE_JORNADA",
    "COLE_MCPIO_UBICACION",
    "COLE_DEPTO_UBICACION",
    "PUNT_LECTURA_CRITICA",
    "PUNT_MATEMATICAS",
    "PUNT_C_NATURALES",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_INGLES",
    "PUNT_GLOBAL",
    "PERCENTIL_GLOBAL"
]

# =========================================================
# CARGAR DATASET
# =========================================================
print("Cargando dataset...")

df = pd.read_csv(INPUT_CSV)

# =========================================================
# FILTRAR COLUMNAS
# =========================================================
columnas_faltantes = [
    col for col in COLUMNAS
    if col not in df.columns
]

if columnas_faltantes:
    raise ValueError(
        f"Columnas faltantes: {columnas_faltantes}"
    )

df = df[COLUMNAS].copy()

# =========================================================
# FAMI_ESTRATOVIVIENDA -> NUMÉRICO
# =========================================================
def transformar_estrato(valor):

    if pd.isna(valor):
        return np.nan

    valor = str(valor).strip().lower()

    if "sin estrato" in valor:
        return 0

    match = re.search(r"\d+", valor)

    if match:
        return int(match.group())

    return np.nan

df["FAMI_ESTRATOVIVIENDA"] = (
    df["FAMI_ESTRATOVIVIENDA"]
    .apply(transformar_estrato)
)

# =========================================================
# SI / NO -> 1 / 0
# =========================================================
MAP_SI_NO = {
    "si": 1,
    "sí": 1,
    "no": 0
}

for col in df.columns:

    valores = (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
    )

    if len(valores) > 0 and set(valores).issubset(
        {"si", "sí", "no"}
    ):

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(MAP_SI_NO)
        )

# =========================================================
# TARGET POR QUINTILES
# =========================================================
df["target"] = pd.qcut(
    df["PUNT_GLOBAL"],
    q=5,
    labels=False
)

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
# FEATURES
# =========================================================
features = [

    col for col in df.columns

    if (

        col.startswith("FAMI_")
        or col.startswith("COLE_")
        or col.startswith("ESTU_")

    )
]

X = df[features].copy()

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
# LIMPIEZA NUMÉRICAS
# =========================================================
for col in num_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )

# =========================================================
# LIMPIEZA CATEGÓRICAS
# =========================================================
for col in cat_cols:

    X[col] = X[col].astype(str)

    X[col] = X[col].replace(
        "nan",
        "missing"
    )

# =========================================================
# RECONSTRUIR DATASET FINAL
# =========================================================
df_final = X.copy()

df_final["target"] = df["target"]

# =========================================================
# GUARDAR
# =========================================================
os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)

df_final.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\nDataset listo para entrenamiento")
print(f"Filas: {len(df_final):,}")
print(f"Columnas: {len(df_final.columns):,}")
print(f"Guardado en: {OUTPUT_CSV}")