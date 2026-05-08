# ============================================================
# K-MEANS + AGRUPAMIENTO JERÁRQUICO AGLOMERATIVO
# ============================================================
# OBJETIVO:
# - Cargar cualquier dataset desde una ruta
# - Detectar automáticamente columnas numéricas
# - Aplicar:
#       ✅ K-Means
#       ✅ Método del Codo
#       ✅ Dendrograma
#       ✅ Líneas de corte
#
# SOLO DEBES CAMBIAR:
#       ruta_dataset
#
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. CARGAR DATASET
# ============================================================

# CAMBIAR ÚNICAMENTE ESTA RUTA
ruta_dataset = "ruta/del/dataset.csv"

# Leer dataset
df = pd.read_csv(ruta_dataset)

# ============================================================
# 2. MOSTRAR INFORMACIÓN
# ============================================================

print("================================================")
print("PRIMERAS FILAS DEL DATASET")
print("================================================")

print(df.head())

print("\n================================================")
print("COLUMNAS DEL DATASET")
print("================================================")

print(df.columns)

# ============================================================
# 3. SELECCIONAR COLUMNAS NUMÉRICAS AUTOMÁTICAMENTE
# ============================================================

X = df.select_dtypes(include=[np.number])

# ============================================================
# 4. ELIMINAR VALORES FALTANTES
# ============================================================

X = X.dropna()

# ============================================================
# 5. VALIDACIÓN
# ============================================================

if X.shape[1] == 0:
    raise ValueError(
        "No se encontraron columnas numéricas en el dataset."
    )

print("\n================================================")
print("COLUMNAS NUMÉRICAS UTILIZADAS")
print("================================================")

print(X.columns.tolist())

print("\n================================================")
print("DATOS UTILIZADOS PARA CLUSTERING")
print("================================================")

print(X.head())

# ============================================================
# 6. ESCALAMIENTO
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# 7. MÉTODO DEL CODO
# ============================================================

inercias = []

# Máximo número de clusters
max_clusters = min(10, len(X))

rango_clusters = range(1, max_clusters + 1)

for k in rango_clusters:

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    modelo.fit(X_scaled)

    inercias.append(modelo.inertia_)

# ============================================================
# 8. GRÁFICA DEL CODO
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    rango_clusters,
    inercias,
    marker='o'
)

plt.title('Método del Codo - K-Means')

plt.xlabel('Número de Clusters')

plt.ylabel('Inercia')

plt.grid(True)

plt.show()

# ============================================================
# 9. K-MEANS FINAL
# ============================================================
# CAMBIAR SEGÚN EL MÉTODO DEL CODO
# ============================================================

k_optimo = 3

modelo_final = KMeans(
    n_clusters=k_optimo,
    random_state=42,
    n_init=10
)

clusters_kmeans = modelo_final.fit_predict(X_scaled)

# Guardar clusters
df.loc[X.index, 'Cluster_KMeans'] = clusters_kmeans

# ============================================================
# 10. RESULTADOS K-MEANS
# ============================================================

print("\n================================================")
print("RESULTADOS K-MEANS")
print("================================================")

print(
    df['Cluster_KMeans']
    .value_counts()
    .sort_index()
)

# ============================================================
# 11. AGRUPAMIENTO JERÁRQUICO
# ============================================================

linked = linkage(
    X_scaled,
    method='ward'
)

# ============================================================
# 12. DENDROGRAMA
# ============================================================

plt.figure(figsize=(15, 8))

dendrogram(
    linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True
)

plt.title(
    'Dendrograma de Agrupamiento Jerárquico (Ward)'
)

plt.xlabel(
    'Índice de Muestra o (Tamaño del Cluster)'
)

plt.ylabel(
    'Distancia Euclidiana'
)

# ============================================================
# 13. LÍNEAS DE CORTE
# ============================================================

# ------------------------------------------------------------
# CORTE PARA 2 CLUSTERS
# ------------------------------------------------------------

k_clusters_2 = 2

clusters_2 = fcluster(
    linked,
    k_clusters_2,
    criterion='maxclust'
)

plt.axhline(
    y=15,
    color='orange',
    linestyle='--',
    label=f'Corte para {k_clusters_2} Clusters'
)

# ------------------------------------------------------------
# CORTE PARA 3 CLUSTERS
# ------------------------------------------------------------

k_clusters_3 = 3

clusters_3 = fcluster(
    linked,
    k_clusters_3,
    criterion='maxclust'
)

plt.axhline(
    y=10,
    color='red',
    linestyle='--',
    label=f'Corte para {k_clusters_3} Clusters'
)

# ------------------------------------------------------------
# CORTE PARA 4 CLUSTERS
# ------------------------------------------------------------

k_clusters_4 = 4

clusters_4 = fcluster(
    linked,
    k_clusters_4,
    criterion='maxclust'
)

plt.axhline(
    y=5,
    color='green',
    linestyle='--',
    label=f'Corte para {k_clusters_4} Clusters'
)

plt.legend()

plt.show()

# ============================================================
# 14. RESULTADOS JERÁRQUICOS
# ============================================================

df.loc[X.index, 'Cluster_Jerarquico'] = clusters_3

print("\n================================================")
print("RESULTADOS JERÁRQUICOS")
print("================================================")

print(
    df['Cluster_Jerarquico']
    .value_counts()
    .sort_index()
)

# ============================================================
# 15. INFORMACIÓN FINAL
# ============================================================

print("\n================================================")
print("RESUMEN")
print("================================================")

print(f"""
Cantidad de registros usados: {len(X)}

Cantidad de variables usadas: {X.shape[1]}

Variables usadas:
{X.columns.tolist()}
""")

# ============================================================
# 16. GUARDAR RESULTADOS
# ============================================================

nombre_salida = "dataset_con_clusters.csv"

df.to_csv(
    nombre_salida,
    index=False
)

print("================================================")
print("ARCHIVO GUARDADO")
print("================================================")

print(f"\nSe guardó el archivo: {nombre_salida}")
