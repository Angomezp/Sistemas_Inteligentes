from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. CARGAR DATASET
# ============================================================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "icfes_transformado.csv")
df = pd.read_csv(DATA_PATH)

# ============================================================
# 2. SELECCIONAR COLUMNAS NUMÉRICAS
# ============================================================
X = df.select_dtypes(include=[np.number])
X = X.dropna()

# ============================================================
# 3. MUESTREO PARA HACER FACTIBLE EL CLUSTERING JERÁRQUICO
# ============================================================
MAX_SAMPLES = 50000   # Ajusta según la memoria disponible
if len(X) > MAX_SAMPLES:
    print(f"Dataset original tiene {len(X)} filas. Se tomará una muestra aleatoria de {MAX_SAMPLES} filas.")
    X = X.sample(n=MAX_SAMPLES, random_state=42)

# ============================================================
# 4. ESCALAMIENTO
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 5. AGRUPAMIENTO JERÁRQUICO
# ============================================================
linked = linkage(X_scaled, method='ward')

# ============================================================
# 6. GRÁFICA DEL CODO (ALTURAS DE FUSIÓN)
# ============================================================
max_clusters = min(20, len(X))
alturas = linked[-max_clusters+1:, 2]

plt.figure(figsize=(8, 5))
plt.plot(range(2, max_clusters+1), alturas, marker='o')
plt.title('Método del Codo - Agrupamiento Jerárquico (Muestra)')
plt.xlabel('Número de Clusters')
plt.ylabel('Distancia de Fusión (Altura)')
plt.grid(True)
plt.show()

# ============================================================
# 7. DENDROGRAMA CON LÍNEAS DE CORTE
# ============================================================
plt.figure(figsize=(15, 8))
dendrogram(
    linked,
    orientation='top',
    distance_sort='descending',
    show_leaf_counts=True
)

# Alturas para 2, 3 y 4 clusters
corte_2clusters = linked[-(2-1), 2]
corte_3clusters = linked[-(3-1), 2]
corte_4clusters = linked[-(4-1), 2]

plt.axhline(y=corte_2clusters, color='orange', linestyle='--',
            label=f'Corte para 2 clusters (altura={corte_2clusters:.2f})')
plt.axhline(y=corte_3clusters, color='red', linestyle='--',
            label=f'Corte para 3 clusters (altura={corte_3clusters:.2f})')
plt.axhline(y=corte_4clusters, color='green', linestyle='--',
            label=f'Corte para 4 clusters (altura={corte_4clusters:.2f})')

plt.title('Dendrograma de Agrupamiento Jerárquico (Ward) - Muestra')
plt.xlabel('Índice de Muestra')
plt.ylabel('Distancia Euclidiana')
plt.legend()
plt.show()

# ============================================================
# 8. ASIGNAR CLUSTERS (ejemplo con 3 clusters)
# ============================================================
k_seleccionado = 3
clusters = fcluster(linked, k_seleccionado, criterion='maxclust')

# Crear un nuevo DataFrame con los clusters de la muestra
df_muestra = X.copy()
df_muestra['Cluster_Jerarquico'] = clusters

# ============================================================
# 9. RESULTADOS
# ============================================================
print("\n================================================")
print("RESULTADOS DEL AGRUPAMIENTO JERÁRQUICO (MUESTRA)")
print("================================================")
print(df_muestra['Cluster_Jerarquico'].value_counts().sort_index())

print("\n================================================")
print("RESUMEN")
print("================================================")
print(f"""
Cantidad de registros usados en clustering: {len(X)}
Cantidad de variables: {X.shape[1]}
Variables:
{X.columns.tolist()}
""")

# ============================================================
# 10. GUARDAR RESULTADOS DE LA MUESTRA
# ============================================================
nombre_salida = "muestra_con_clusters_jerarquico.csv"
df_muestra.to_csv(nombre_salida, index=False)

print("================================================")
print("ARCHIVO GUARDADO")
print("================================================")
print(f"\nSe guardó el archivo: {nombre_salida}")