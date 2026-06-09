import pandas as pd

# Configuración de rutas

INPUT_CSV = "CSV/icfes_transformado.csv"
#OUTPUT_STATS = "Resultados/Analisis_Basico/AnalisisBasico_12Grupos.txt"
OUTPUT_STATS = "Resultados/Analisis_Basico/AnalisisBasico_Quintiles.txt"

COLUMNA = "PUNT_GLOBAL"


# Función para categorizar en 12 grupos de igual longitud 

# def categorizar(x):
#     x = float(x)

#     if x <= 41.7:
#         return 0

#     elif x <= 83.4:
#         return 1

#     elif x <= 125.1:
#         return 2

#     elif x <= 166.8:
#         return 3

#     elif x <= 208.5:
#         return 4

#     elif x <= 250.2:
#         return 5

#     elif x <= 291.9:
#         return 6

#     elif x <= 333.6:
#         return 7

#     elif x <= 375.3:
#         return 8

#     elif x <= 417:
#         return 9

#     elif x <= 458.7:
#         return 10

#     else:
#         return 11

# leer el dataset
df = pd.read_csv(INPUT_CSV)

# Eliminar nulos en la columna de interés
df = df[df[COLUMNA].notna()].copy()

# Convertir a numérico
df[COLUMNA] = pd.to_numeric(df[COLUMNA], errors="coerce")
df = df[df[COLUMNA].notna()]

# Estadísticas básicas

stats = df[COLUMNA].describe()

# Medidas adicionales
varianza = df[COLUMNA].var()
asimetria = df[COLUMNA].skew()
curtosis = df[COLUMNA].kurt()

# Categorizar

#df["CATEGORIA"] = df[COLUMNA].apply(categorizar)
df["CATEGORIA"] = pd.qcut(
    df["PUNT_GLOBAL"],
    q=5,
    labels=False
)

conteo_categorias = (
    df["CATEGORIA"]
    .value_counts()
    .sort_index()
)

# Generar reporte

reporte = []

reporte.append("===== ESTADÍSTICAS BÁSICAS =====\n")
reporte.append(stats.to_string())
reporte.append("\n")

reporte.append(f"\nVarianza: {varianza:.4f}")
reporte.append(f"\nAsimetría: {asimetria:.4f}")
reporte.append(f"\nCurtosis: {curtosis:.4f}\n")

reporte.append("\n===== CONTEO POR CATEGORÍA =====\n")

for categoria, cantidad in conteo_categorias.items():
    porcentaje = (cantidad / len(df)) * 100
    reporte.append(
        f"Categoría {categoria:>2}: "
        f"{cantidad:>10} registros "
        f"({porcentaje:.2f}%)"
    )

texto_reporte = "\n".join(reporte)

# Mostrar reporte en consola

print(texto_reporte)

# Guardar reporte en archivo de texto

with open(OUTPUT_STATS, "w", encoding="utf-8") as f:
    f.write(texto_reporte)

print(f"\nReporte guardado en: {OUTPUT_STATS}")