import os
import pandas as pd
import numpy as np
import re

# =========================================================
# RUTAS
# =========================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "icfes.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "icfes_transformado.csv")

# =========================================================
# CARGAR
# =========================================================
df = pd.read_csv(INPUT_PATH)

# =========================================================
# 1. FAMI_ESTRATOVIVIENDA → número
# =========================================================
def transformar_estrato(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    if "sin estrato" in x:
        return 0

    # extraer número (ej: "estrato 3")
    match = re.search(r"\d+", x)
    if match:
        return int(match.group())

    return np.nan

df["FAMI_ESTRATOVIVIENDA"] = df["FAMI_ESTRATOVIVIENDA"].apply(transformar_estrato)

# =========================================================
# 2. SI / NO → 1 / 0
# =========================================================
map_si_no = {
    "si": 1,
    "sí": 1,
    "no": 0
}

def transformar_binaria(col):
    return col.astype(str).str.strip().str.lower().map(map_si_no)

# detectar columnas tipo si/no automáticamente
for col in df.columns:
    valores = df[col].dropna().astype(str).str.strip().str.lower().unique()

    if set(valores).issubset({"si", "sí", "no"}):
        df[col] = transformar_binaria(df[col])

# =========================================================
# VERIFICACIÓN
# =========================================================
print("\nTipos finales:\n")
print(df.dtypes)

# =========================================================
# GUARDAR
# =========================================================
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nDataset transformado guardado en: {OUTPUT_PATH}")