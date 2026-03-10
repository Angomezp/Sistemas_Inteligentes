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
    # Definir formas de las piezas (igual que en encontrar_mejor_posicion)
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

    # Crear copia de la matriz
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
        return None, 0  # No hay espacio

    # Colocar la pieza
    for i in range(altura_p):
        for j in range(ancho_p):
            if forma[i][j] == 1:
                nueva[fila_caida + i][col + j] = 1

    # Eliminar líneas completas
    nueva, lineas = eliminar_lineas(nueva)
    return nueva, lineas

def puntuar_tablero(matriz):
    """
    Evalúa la bondad del tablero. Mayor puntuación es mejor.
    Factores: menos huecos, menos altura, menos irregularidad, líneas recientes ya contadas.
    """
    alturas = alturas_columna(matriz)
    huecos = contar_huecos(matriz)
    bump = calcular_bumpiness(alturas)
    altura_max = max(alturas) if alturas else 0

    # Pesos (ajustables)
    PESO_HUECOS = -10
    PESO_ALTURA = -1
    PESO_BUMP = -0.5
    PESO_LINEAS = 100  # Las líneas ya se reflejan en la matriz, pero queremos dar mucho peso a que haya menos altura/huecos después de líneas

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
    # Definir formas (mismo diccionario que antes)
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
            # Simular colocación
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero)
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot

########################################
# BOT MEJORADO (solo esta función se modifica)
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

            # Actualizar estado del hold (lo que realmente vemos en pantalla)
            if piezas_hold:
                pieza_en_hold = piezas_hold[0][4]
            else:
                pieza_en_hold = None

            # Ordenar piezas siguientes por posición Y (de arriba a abajo = más próximas)
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
                pieza_actual = piezas_spawn[0][4]  # El tipo de pieza
                print(f"\n>>> NUEVA PIEZA DETECTADA: {pieza_actual}")

                # Obtener próximas piezas en orden (la primera es la más próxima)
                proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []

                # Evaluar las opciones
                # Opción 1: Colocar la pieza actual ahora
                punt_actual, col_actual, rot_actual = mejor_placement(matriz_tablero, pieza_actual)
                puede_colocar_actual = col_actual is not None

                # Opción 2: Usar hold (guardar o intercambiar)
                opcion_hold = None  # 'guardar' o 'swap'
                punt_hold = float('-inf')
                if not ultimo_hold_usado:
                    if pieza_en_hold is None:
                        # Simular guardar la actual y luego colocar la siguiente (si existe)
                        if proximas_piezas:
                            # Simular tablero actual (no cambia porque guardar no coloca)
                            # Luego colocar la primera siguiente
                            punt_sig, col_sig, rot_sig = mejor_placement(matriz_tablero, proximas_piezas[0])
                            if col_sig is not None:
                                opcion_hold = 'guardar'
                                # Bonus por guardar una pieza problemática? Se refleja en la puntuación de la siguiente
                                punt_hold = punt_sig
                    else:
                        # Simular intercambiar: la actual va al hold, la del hold sale
                        # Colocar la pieza que estaba en hold
                        punt_swap, col_swap, rot_swap = mejor_placement(matriz_tablero, pieza_en_hold)
                        if col_swap is not None:
                            opcion_hold = 'swap'
                            punt_hold = punt_swap

                # Decidir acción: la que dé mayor puntuación
                if puede_colocar_actual and punt_actual >= punt_hold:
                    # Colocar pieza actual
                    print(f"Colocando {pieza_actual} (puntuación: {punt_actual:.1f})")
                    exito = colocar_pieza_mejorada(pieza_actual, matriz_tablero, keyboard, col_actual, rot_actual)
                    if exito:
                        print(f"Pieza {pieza_actual} colocada")
                        ultimo_hold_usado = False
                    else:
                        print(f"Error colocando {pieza_actual}")
                        contador_fallos += 1
                    pieza_actual = None

                elif punt_hold > float('-inf') and punt_hold > punt_actual:
                    # Ejecutar acción de hold
                    if opcion_hold == 'guardar':
                        print(f"Guardando {pieza_actual} en hold (siguiente {proximas_piezas[0]} puntúa {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        # La pieza actual se guarda, esperar a que aparezca la siguiente
                        pieza_actual = None
                        time.sleep(0.3)
                        # Nota: la pieza en hold ahora es la actual, pero no la asignamos aún porque vendrá en spawn
                        # Simplemente esperamos a que la nueva pieza aparezca
                    elif opcion_hold == 'swap':
                        print(f"Intercambiando: {pieza_actual} con {pieza_en_hold} (puntuación {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        # Después del swap, la pieza que estaba en hold ahora está en spawn
                        # Debemos esperar a que aparezca y luego colocarla
                        time.sleep(0.3)
                        # La nueva pieza actual será la que vino del hold, pero la detectaremos en la siguiente iteración
                        pieza_actual = None
                    else:
                        # No debería ocurrir
                        print("Error en opción de hold")
                        pieza_actual = None
                else:
                    # No se puede colocar ni usar hold
                    print(f"No se puede colocar {pieza_actual} y hold no disponible o no mejora")
                    # Intentar colocar de todas formas (por si acaso)
                    if puede_colocar_actual:
                        exito = colocar_pieza_mejorada(pieza_actual, matriz_tablero, keyboard, col_actual, rot_actual)
                        if exito:
                            print(f"Pieza {pieza_actual} colocada (forzada)")
                        else:
                            contador_fallos += 1
                    else:
                        print(f"No hay lugar para {pieza_actual}, esperando...")
                    pieza_actual = None

            # Pequeña pausa para no saturar
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nBot detenido")
    except Exception as e:
        print(f"Error en el bot: {e}")
        import traceback
        traceback.print_exc()

def colocar_pieza_mejorada(pieza, matriz_tablero, keyboard, col_objetivo, rot_objetivo):
    """
    Coloca la pieza en la posición y rotación específicas.
    Utiliza el mismo sistema de teclado que el original.
    """
    print(f"Colocando pieza {pieza} en columna {col_objetivo} con rotación {rot_objetivo}")

    # Asumimos spawn en columna 4 (índice 4) para piezas de ancho 1-4.
    # Pero el spawn real puede variar. Tomamos como referencia col_actual = 4.
    col_actual = 4
    desplazamiento = col_objetivo - col_actual

    # Aplicar rotaciones necesarias
    if rot_objetivo > 0:
        print(f"Aplicando {rot_objetivo} rotación(es)")
        for _ in range(rot_objetivo):
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

    # Bajar la pieza inmediatamente
    print("Soltando pieza...")
    time.sleep(0.1)
    keyboard.press(Key.space)
    time.sleep(0.05)
    keyboard.release(Key.space)

    return True
