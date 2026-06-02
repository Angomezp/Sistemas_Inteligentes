from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
# 2. SELECCIONAR COLUMNAS NUMÉRICAS Y LIMPIAR
# ============================================================
X = df.select_dtypes(include=[np.number]).dropna()

# ============================================================
# 3. MUESTREO PARA HACER FACTIBLES AMBOS MÉTODOS
# ============================================================
MAX_SAMPLES = 2000000   # Para jerárquico; K‑Means puede usar más, pero seguimos la misma muestra
if len(X) > MAX_SAMPLES:
    print(f"Dataset original con {len(X)} filas. Se toma una muestra de {MAX_SAMPLES} para ambos análisis.")
    X_sample = X.sample(n=MAX_SAMPLES, random_state=42)
else:
    X_sample = X

# ============================================================
# 4. ESCALAMIENTO
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# ============================================================
# 6. K-MEANS CON GRÁFICA DEL CODO (INERCIA)
# ============================================================
inercias = []
k_range = range(1, min(15, len(X_sample)))   # Hasta 10 clusters
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
plt.show()

# (Opcional) Mostrar el punto de inflexión automático
# Se puede calcular la diferencia de inercias para sugerir un k
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