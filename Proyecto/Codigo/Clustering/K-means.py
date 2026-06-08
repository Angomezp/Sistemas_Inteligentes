from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Path
DATA_PATH = "CSV/icfes_transformado.csv"
df = pd.read_csv(DATA_PATH)

# Seleccionar solo las columnas numéricas para el clustering
X = df.select_dtypes(include=[np.number]).dropna()

X.info()  # Verificar número de filas y columnas después de limpiar

# Limitar el número de muestras para evitar problemas de memoria
MAX_SAMPLES = 2_000_000   
if len(X) > MAX_SAMPLES:
    print(f"Dataset original con {len(X)} filas. Se toma una muestra de {MAX_SAMPLES} para ambos análisis.")
    X_sample = X.sample(n=MAX_SAMPLES, random_state=42)
else:
    X_sample = X

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Graficas de codo para K-Means
inercias = []
k_range = range(1, min(15, len(X_sample)))   
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(k_range, inercias, marker='o')
plt.title('Método del Codo - K-Means')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.grid(True)
os.makedirs("Imagenes", exist_ok=True)
plt.savefig("Imagenes/KMeans_Codo.png", bbox_inches='tight', dpi=150)
plt.show()
plt.close()

# Calcular la segunda derivada para sugerir un número de clusters
diferencias = np.diff(inercias, 2)  # segunda derivada aproximada
sugerido_k = np.argmin(diferencias) + 2 if len(diferencias) > 0 else 2
print(f"Posible número de clusters según el codo de K-Means: {sugerido_k}")


print("\n================================================")
print("RESUMEN")
print("================================================")
print(f"""
Registros procesados: {len(X_sample)}
Variables usadas: {X_sample.shape[1]}
K-Means óptimo sugerido: {sugerido_k} (ver gráfica)
""")