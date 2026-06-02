import pandas as pd

# Configuración
INPUT_CSV = "CSV/icfes_original.csv"
OUTPUT_CSV = "CSV/icfes_transformado.csv"

# Columnas que vamos a usar
COLUMNAS = [
    'ESTU_DEPTO_RESIDE', 'ESTU_MCPIO_RESIDE', 'FAMI_ESTRATOVIVIENDA',
    'FAMI_PERSONASHOGAR', 'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE',
    'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR',
    'ESTU_DEDICACIONLECTURADIARIA', 'ESTU_DEDICACIONINTERNET',
    'ESTU_HORASSEMANATRABAJA', 'COLE_NATURALEZA', 'COLE_BILINGUE',
    'COLE_JORNADA', 'COLE_MCPIO_UBICACION', 'COLE_DEPTO_UBICACION',
    'PUNT_LECTURA_CRITICA', 'PUNT_MATEMATICAS', 'PUNT_C_NATURALES',
    'PUNT_SOCIALES_CIUDADANAS', 'PUNT_INGLES', 'PUNT_GLOBAL',
    'PERCENTIL_GLOBAL'
]

# Leer archivo
df = pd.read_csv(INPUT_CSV)

# Verificar que todas las columnas existan
columnas_faltantes = [col for col in COLUMNAS if col not in df.columns]

if columnas_faltantes:
    raise ValueError(
        f"Las siguientes columnas no existen en el CSV: {columnas_faltantes}"
    )

# Filtrar columnas
df_filtrado = df[COLUMNAS]

# Guardar resultado
df_filtrado.to_csv(OUTPUT_CSV, index=False)

print(f"Archivo guardado como: {OUTPUT_CSV}")