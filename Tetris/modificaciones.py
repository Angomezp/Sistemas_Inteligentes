# En ejecutar_bot, dentro del if piezas_spawn and pieza_actual is None:

if piezas_spawn and pieza_actual is None:
    pieza_actual = piezas_spawn[0][4]
    # Obtener bounding box de la pieza en spawn (coordenadas relativas a la imagen de spawn)
    x, y, w, h, _ = piezas_spawn[0]
    # Calcular columna actual aproximada
    cell_w, _ = obtener_tamano_celda(t)  # ya tenemos cell_w de antes, pero podemos recalcular
    col_actual = int(round(x / cell_w))
    # Asegurar que esté en rango
    col_actual = max(0, min(col_actual, BOARD_COLS - 1))
    print(f"Pieza detectada en columna aproximada: {col_actual}")

    # ... luego en la decisión de colocar:
    if puede_colocar_actual and punt_actual >= punt_hold:
        print(f"Colocando {pieza_actual} (puntuación: {punt_actual:.1f})")
        exito = colocar_pieza_mejorada(pieza_actual, matriz_tablero, keyboard, col_actual, col_actual, rot_actual)
        # Nota: pasamos col_actual dos veces? En realidad necesitamos col_objetivo y col_actual.
        # La función debe recibir col_actual (posición actual) y col_objetivo.
        # Así que la llamada sería:
        exito = colocar_pieza_mejorada(pieza_actual, matriz_tablero, keyboard, col_actual, col_objetivo, rot_objetivo)

  ##############################En colocar pieza mejorada
def colocar_pieza_mejorada(pieza, matriz_tablero, keyboard, col_actual, col_objetivo, rot_objetivo):
    """
    Coloca la pieza en la posición y rotación específicas.
    col_actual: columna actual de la esquina izquierda de la pieza (en la rotación inicial).
    col_objetivo: columna deseada para la esquina izquierda en la rotación final.
    rot_objetivo: rotación deseada (0..n-1)
    """
    # Definir formas de las piezas (mismo diccionario que en mejor_placement)
    formas = {
        'I': [
            [[1,1,1,1]],
            [[1],[1],[1],[1]]
        ],
        'O': [
            [[1,1],[1,1]]
        ],
        'T': [
            [[0,1,0],[1,1,1]],
            [[1,0],[1,1],[1,0]],
            [[1,1,1],[0,1,0]],
            [[0,1],[1,1],[0,1]]
        ],
        'S': [
            [[0,1,1],[1,1,0]],
            [[1,0],[1,1],[0,1]]
        ],
        'Z': [
            [[1,1,0],[0,1,1]],
            [[0,1],[1,1],[1,0]]
        ],
        'J': [
            [[1,0,0],[1,1,1]],
            [[1,1],[1,0],[1,0]],
            [[1,1,1],[0,0,1]],
            [[0,1],[0,1],[1,1]]
        ],
        'L': [
            [[0,0,1],[1,1,1]],
            [[1,0],[1,0],[1,1]],
            [[1,1,1],[1,0,0]],
            [[1,1],[0,1],[0,1]]
        ]
    }
    
    # Número de rotaciones posibles para esta pieza
    num_rot = len(formas[pieza])
    
    # Rotación inicial asumida 0
    rot_inicial = 0
    rot_necesarias = (rot_objetivo - rot_inicial) % num_rot
    
    # Desplazamiento horizontal necesario
    desplazamiento = col_objetivo - col_actual
    
    print(f"Colocando {pieza}: desde col {col_actual} a col {col_objetivo}, rot necesarias={rot_necesarias}, desplazamiento={desplazamiento}")
    
    # Decidir orden: primero rotar o primero mover? 
    # Por simplicidad, primero rotamos y luego movemos.
    # Pero ten en cuenta que la rotación puede cambiar la columna de la esquina.
    # Si el juego rota alrededor del centro, la esquina se desplazará.
    # Para compensar, podríamos necesitar ajustar el desplazamiento.
    # Por ahora, asumimos que la rotación no cambia la columna de la esquina (simplificación).
    
    # Aplicar rotaciones
    if rot_necesarias > 0:
        print(f"Aplicando {rot_necesarias} rotación(es)")
        for _ in range(rot_necesarias):
            keyboard.press(Key.up)
            time.sleep(0.05)
            keyboard.release(Key.up)
            time.sleep(0.05)
    
    # Mover horizontalmente
    if desplazamiento > 0:
        print(f"Moviendo derecha {desplazamiento} veces")
        for _ in range(desplazamiento):
            keyboard.press(Key.right)
            time.sleep(0.05)
            keyboard.release(Key.right)
            time.sleep(0.05)
    elif desplazamiento < 0:
        print(f"Moviendo izquierda {abs(desplazamiento)} veces")
        for _ in range(abs(desplazamiento)):
            keyboard.press(Key.left)
            time.sleep(0.05)
            keyboard.release(Key.left)
            time.sleep(0.05)
    
    # Soltar la pieza
    print("Soltando pieza...")
    time.sleep(0.1)
    keyboard.press(Key.space)
    time.sleep(0.05)
    keyboard.release(Key.space)
    
    return True
  
