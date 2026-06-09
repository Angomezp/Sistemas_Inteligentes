Proyecto — Resumen rápido

Estructura y propósito
- `CSV/`: Datos de entrada y salida
- `Codigo/`: scripts organizados por propósito
  - `Preprocesamiento/`
    - `Filtrado.py`: limpia y transforma `icfes_original.csv`, genera `icfes_transformado.csv`
    - `AnalisisBasico.py`: análisis exploratorio y estadísticas básicas
  - `CatBoost/`
    - `CatBoost-4Params.py`: entrenamiento/ajuste con CatBoost (usa `icfes_transformado.csv`)
  - `RandomForest/`
    - `RandomForest-4Params.py`: entrenamiento/ajuste con RandomForest
  - `XGBoost/`
    - `XGBoost-4Params.py`: entrenamiento/ajuste con XGBoost
  - `Clustering/`
    - `clusters.py`, `K-means.py`: scripts de clustering y pruebas de k-means
- `Imagenes/`: imágenes usadas en análisis o reports
- `Resultados/`: outputs y métricas por modelo (subcarpetas por método)
- `Proyecto_3-Reporte.pdf`: Reporte final del proyecto.
- `Proyecto_3-Tablas.pdf`: Tablas y gráficas. (Mejor visualización)

Requisitos
- Python 3.8+ (probado con 3.13 en este entorno)
- Dependencias listadas en `requirements.txt` (ver sección Instalación)

Instalación (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Comandos de uso (desde la carpeta `Proyecto`)
- Preprocesar y generar el CSV transformado:
```powershell
python Codigo\Preprocesamiento\Filtrado.py
```
- Ejecutar clustering:
```powershell
python Codigo\Clustering\clusters.py
python Codigo\Clustering\K-means.py
```
- Ejecutar análisis básico:
```powershell
python Codigo\Preprocesamiento\AnalisisBasico.py
```
- Entrenar modelos (ejemplos):
```powershell
python Codigo\CatBoost\CatBoost-4Params.py
python Codigo\RandomForest\RandomForest-4Params.py
python Codigo\XGBoost\XGBoost-4Params.py
```

Notas
- `Filtrado.py` escribe en `Proyecto/CSV/icfes_transformado.csv` y espera `icfes_original.csv` en `Proyecto/CSV/`.
- Si faltan paquetes usa `pip install -r requirements.txt`.
