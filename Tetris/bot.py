########################################
# FUNCIONES AUXILIARES PARA EL BOT MEJORADO
########################################

def alturas_columna(matriz):
    """Calcula la altura de cada columna (fila desde arriba donde hay un bloque)"""
    alturas = []
    for col in range(BOARD_COLS):
        for fila in range(BOARD_ROWS):
            if matriz[fila][col] == 1:
                alturas.append(BOARD_ROWS - fila)
                break
        else:
            alturas.append(0)
    return alturas

def contar_huecos(matriz):
    """Cuenta los huecos (celdas vacías con al menos un bloque encima)"""
    huecos = 0
    for col in range(BOARD_COLS):
        bloque_encima = False
        for fila in range(BOARD_ROWS):
            if matriz[fila][col] == 1:
                bloque_encima = True
            elif bloque_encima and matriz[fila][col] == 0:
                huecos += 1
    return huecos

def calcular_bumpiness(alturas):
    """Mide la irregularidad del tablero (suma de diferencias entre columnas adyacentes)"""
    bump = 0
    for i in range(len(alturas)-1):
        bump += abs(alturas[i] - alturas[i+1])
    return bump

def eliminar_lineas(matriz):
    """Elimina filas completas y devuelve nueva matriz y número de líneas eliminadas"""
    nuevas_filas = []
    lineas = 0
    for fila in range(BOARD_ROWS):
        if all(matriz[fila][col] == 1 for col in range(BOARD_COLS)):
            lineas += 1
        else:
            nuevas_filas.append(matriz[fila].copy())
    # Añadir filas vacías arriba
    nuevas_filas = [[0]*BOARD_COLS for _ in range(lineas)] + nuevas_filas
    return np.array(nuevas_filas, dtype=int), lineas

def simular_placement(matriz, pieza, rot, col):
    """
    Simula colocar una pieza en la matriz y devuelve el nuevo tablero y líneas eliminadas.
    pieza: tipo de pieza (str)
    rot: índice de rotación (0..n)
    col: columna donde colocar (esquina superior izquierda de la pieza)
    """
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
    forma = formas[pieza][rot]
    altura_p = len(forma)
    ancho_p = len(forma[0])

    nueva = matriz.copy()

    # Encontrar fila de caída
    fila_caida = None
    for fila in range(BOARD_ROWS - altura_p + 1):
        colision = False
        for i in range(altura_p):
            for j in range(ancho_p):
                if forma[i][j] == 1:
                    if nueva[fila + i][col + j] == 1:
                        colision = True
                        break
            if colision:
                break
        if colision:
            fila_caida = fila - 1
            break
    if fila_caida is None:
        fila_caida = BOARD_ROWS - altura_p

    if fila_caida < 0:
        return None, 0

    # Colocar la pieza
    for i in range(altura_p):
        for j in range(ancho_p):
            if forma[i][j] == 1:
                nueva[fila_caida + i][col + j] = 1

    nueva, lineas = eliminar_lineas(nueva)
    return nueva, lineas

def puntuar_tablero(matriz):
    """
    Evalúa la bondad del tablero. Mayor puntuación es mejor.
    Factores: menos huecos, menos altura, menos irregularidad.
    """
    alturas = alturas_columna(matriz)
    huecos = contar_huecos(matriz)
    bump = calcular_bumpiness(alturas)
    altura_max = max(alturas) if alturas else 0

    PESO_HUECOS = -10
    PESO_ALTURA = -1
    PESO_BUMP = -0.5

    puntuacion = (PESO_HUECOS * huecos +
                  PESO_ALTURA * altura_max +
                  PESO_BUMP * bump)
    return puntuacion

def mejor_placement(matriz, pieza):
    """
    Encuentra la mejor posición (rotación y columna) para una pieza.
    Devuelve (mejor_puntuacion, mejor_col, mejor_rot)
    Si no hay placement posible, devuelve (-inf, None, None)
    """
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
    if pieza not in formas:
        return float('-inf'), None, None

    rotaciones = formas[pieza]
    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None

    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        for col in range(BOARD_COLS - ancho_p + 1):
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero)
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot

def colocar_pieza_mejorada(pieza, columna_spawn_inicial, columna_objetivo, rotacion_objetivo, keyboard, spawn_region, cell_w):
    """
    Coloca la pieza desde su posición actual de spawn hasta la posición objetivo.
    Primero rota, luego detecta la nueva posición y mueve horizontalmente.
    """
    # Definir formas para obtener número de rotaciones
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
    
    num_rot = len(formas[pieza])
    rot_inicial = 0
    rot_necesarias = (rotacion_objetivo - rot_inicial) % num_rot
    
    print(f"Rotando {rot_necesarias} veces desde rotación inicial")
    
    # Aplicar rotaciones
    for _ in range(rot_necesarias):
        keyboard.press(Key.up)
        time.sleep(0.05)
        keyboard.release(Key.up)
        time.sleep(0.1)  # esperar a que la pieza se estabilice
    
    # Después de rotar, capturar de nuevo la zona spawn para obtener la nueva columna
    spawn_img = capture_screen(spawn_region)
    piezas_detectadas = detectar_piezas_spawn(spawn_img)
    
    if not piezas_detectadas:
        print("Error: No se pudo detectar la pieza después de rotar. Usando columna inicial.")
        columna_actual = columna_spawn_inicial
    else:
        # Tomar la primera pieza detectada (debería ser la misma)
        x_spawn = piezas_detectadas[0][0]
        columna_actual = int(round(x_spawn / cell_w))
        columna_actual = max(0, min(columna_actual, BOARD_COLS-1))
        print(f"Nueva posición después de rotar: columna {columna_actual}")
    
    desplazamiento = columna_objetivo - columna_actual
    print(f"Moviendo desde col {columna_actual} a col {columna_objetivo} (desp={desplazamiento})")
    
    # Mover horizontalmente
    if desplazamiento > 0:
        for _ in range(desplazamiento):
            keyboard.press(Key.right)
            time.sleep(0.05)
            keyboard.release(Key.right)
            time.sleep(0.05)
    elif desplazamiento < 0:
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

########################################
# BOT MEJORADO
########################################

def ejecutar_bot():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado (modo Tetris inteligente)")
    print("Presiona Ctrl+C para detener")

    # Variables de estado
    pieza_actual = None
    pieza_en_hold = None
    pieza_guardada_hold = None  # Para tracking interno
    ultimo_hold_usado = False
    contador_fallos = 0

    try:
        while True:
            # Capturar todas las zonas
            t = config["tablero"]
            s = config["siguientes"]
            h = config["hold"]

            # Calcular zona spawn
            cell_w, cell_h = obtener_tamano_celda(t)
            spawn_height = int(cell_h * SPAWN_ROWS)

            spawn_region = (
                t[0],
                t[1] - spawn_height,
                t[2],
                spawn_height
            )

            # Capturar imágenes
            tablero_img = capture_screen(t)
            spawn_img = capture_screen(spawn_region)
            siguientes_img = capture_screen(s)
            hold_img = capture_screen(h)

            # Obtener matriz del tablero
            matriz_tablero = obtener_matriz_tablero(tablero_img, t)

            # Detectar piezas
            piezas_spawn = detectar_piezas_spawn(spawn_img)
            piezas_siguientes = detectar_piezas_next_hold(siguientes_img)
            piezas_hold = detectar_piezas_next_hold(hold_img)

            # Actualizar estado del hold
            if piezas_hold:
                pieza_en_hold = piezas_hold[0][4]
            else:
                pieza_en_hold = None

            # Ordenar piezas siguientes por posición Y (más próximas primero)
            if piezas_siguientes:
                piezas_siguientes.sort(key=lambda p: p[1])

            # Mostrar información de debug
            if piezas_spawn:
                print(f"\nPieza en spawn: {piezas_spawn[0][4]}")
            if piezas_siguientes:
                print(f"Siguientes (en orden): {[p[4] for p in piezas_siguientes[:4]]}")
            if pieza_en_hold:
                print(f"Hold: {pieza_en_hold}")

            # Si hay pieza en spawn y no hay pieza actual, es nueva pieza
            if piezas_spawn and pieza_actual is None:
                pieza_actual = piezas_spawn[0][4]
                print(f"\n>>> NUEVA PIEZA DETECTADA: {pieza_actual}")

                # Calcular columna actual de la pieza en spawn
                x_spawn = piezas_spawn[0][0]
                columna_spawn = int(round(x_spawn / cell_w))
                columna_spawn = max(0, min(columna_spawn, BOARD_COLS-1))
                print(f"Posición detectada: columna {columna_spawn} (x={x_spawn}px, cell_w={cell_w:.1f})")

                # Obtener próximas piezas
                proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []

                # Evaluar opciones
                punt_actual, col_objetivo, rot_objetivo = mejor_placement(matriz_tablero, pieza_actual)
                puede_colocar_actual = col_objetivo is not None

                # Opción de hold
                opcion_hold = None
                punt_hold = float('-inf')
                if not ultimo_hold_usado:
                    if pieza_en_hold is None:
                        if proximas_piezas:
                            punt_sig, col_sig, rot_sig = mejor_placement(matriz_tablero, proximas_piezas[0])
                            if col_sig is not None:
                                opcion_hold = 'guardar'
                                punt_hold = punt_sig
                    else:
                        punt_swap, col_swap, rot_swap = mejor_placement(matriz_tablero, pieza_en_hold)
                        if col_swap is not None:
                            opcion_hold = 'swap'
                            punt_hold = punt_swap

                # Decidir acción
                if puede_colocar_actual and punt_actual >= punt_hold:
                    print(f"Colocando {pieza_actual} (puntuación: {punt_actual:.1f})")
                    exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                    keyboard, spawn_region, cell_w)
                    if exito:
                        print(f"Pieza {pieza_actual} colocada")
                        ultimo_hold_usado = False
                    else:
                        print(f"Error colocando {pieza_actual}")
                        contador_fallos += 1
                    pieza_actual = None

                elif punt_hold > float('-inf') and punt_hold > punt_actual:
                    if opcion_hold == 'guardar':
                        print(f"Guardando {pieza_actual} en hold (siguiente {proximas_piezas[0]} puntúa {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        pieza_actual = None
                        time.sleep(0.3)
                    elif opcion_hold == 'swap':
                        print(f"Intercambiando: {pieza_actual} con {pieza_en_hold} (puntuación {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        time.sleep(0.3)
                        pieza_actual = None
                    else:
                        print("Error en opción de hold")
                        pieza_actual = None
                else:
                    print(f"No se puede colocar {pieza_actual} y hold no disponible o no mejora")
                    if puede_colocar_actual:
                        exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                        keyboard, spawn_region, cell_w)
                        if exito:
                            print(f"Pieza {pieza_actual} colocada (forzada)")
                        else:
                            contador_fallos += 1
                    else:
                        print(f"No hay lugar para {pieza_actual}, esperando...")
                    pieza_actual = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nBot detenido")
    except Exception as e:
        print(f"Error en el bot: {e}")
        import traceback
        traceback.print_exc()
