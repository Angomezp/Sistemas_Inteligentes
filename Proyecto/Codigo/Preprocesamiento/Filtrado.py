import os
import re
import numpy as np
import pandas as pd

# Paths
# La carpeta CSV está en el nivel de `Proyecto/CSV`, dos niveles arriba de este archivo
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

INPUT_CSV = os.path.join(BASE_DIR, "CSV", "icfes_original.csv")

OUTPUT_CSV = os.path.join(BASE_DIR, "CSV", "icfes_transformado.csv")

# Columnas a mantener
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

# Cargar dataset
print("Cargando dataset...")

df = pd.read_csv(INPUT_CSV)

# Filtar columnas
columnas_faltantes = [
    col for col in COLUMNAS
    if col not in df.columns
]

if columnas_faltantes:
    raise ValueError(
        f"Columnas faltantes: {columnas_faltantes}"
    )

df = df[COLUMNAS].copy()

# Transformar estrato de categórico a numérico
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

# Mapear "si"/"no" a 1/0
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

# Crear target por quintiles del puntaje global
df["target"] = pd.qcut(
    df["PUNT_GLOBAL"],
    q=5,
    labels=False
)

# Convertir columnas numéricas a tipo numérico, dejando las no numéricas sin cambios
df = df.apply(
    lambda col: pd.to_numeric(
        col,
        errors="ignore"
    )
)

# Features: solo las que empiezan con FAMI_, COLE_ o ESTU_, excluyendo las que empiezan con PUNT_ o PERCENTIL_
features = [

    col for col in df.columns

    if (
        col.startswith("FAMI_")
        or col.startswith("COLE_")
        or col.startswith("ESTU_")
    )
]

X = df[features].copy()

# Columnas numéricas y categóricas
num_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

cat_cols = X.select_dtypes(
    include=["object", "category"]
).columns

# Limpieza numéricas
for col in num_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )

    X[col] = X[col].fillna(
        X[col].median()
    )

# Limpieza categóricas
for col in cat_cols:

    X[col] = X[col].astype(str)

    X[col] = X[col].replace(
        "nan",
        "missing"
    )

# Dataset final
df_final = X.copy()

df_final["target"] = df["target"]

# Guardar dataset limpio
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