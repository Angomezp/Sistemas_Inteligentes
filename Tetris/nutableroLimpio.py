####################Puntuar tablero
def puntuar_tablero(matriz, lineas_eliminadas=0):
    """
    Evalúa la bondad del tablero. Mayor puntuación es mejor.
    Factores: líneas eliminadas (muy positivo), menos huecos, menos altura, menos irregularidad.
    Si el tablero está completamente vacío (altura máxima 0), se da una recompensa enorme.
    """
    alturas = alturas_columna(matriz)
    huecos = contar_huecos(matriz)
    bump = calcular_bumpiness(alturas)
    altura_max = max(alturas) if alturas else 0

    # Recompensa masiva si el tablero queda limpio
    if altura_max == 0:
        return 10000

    # Pesos
    PESO_LINEAS = 500      # por cada línea eliminada
    PESO_HUECOS = -25
    PESO_ALTURA = -10
    PESO_BUMP = -2

    # Bonus especial por Tetris (4 líneas) y triple (3 líneas)
    if lineas_eliminadas == 4:
        bonus = 2000
    elif lineas_eliminadas == 3:
        bonus = 1000
    else:
        bonus = 0

    puntuacion = (PESO_LINEAS * lineas_eliminadas +
                  PESO_HUECOS * huecos +
                  PESO_ALTURA * altura_max +
                  PESO_BUMP * bump +
                  bonus)
    return puntuacion
########Mejor Placemn t
    def mejor_placement(matriz, pieza):
    """
    Estrategia completa (robusta) - evalúa todas las posiciones
    Encuentra la mejor posición (rotación y columna) para una pieza.
    Devuelve (mejor_puntuacion, mejor_col, mejor_rot)
    Si no hay placement posible, devuelve (-inf, None, None)
    """
    if pieza not in FORMAS_PIEZAS:
        return float('-inf'), None, None

    rotaciones = FORMAS_PIEZAS[pieza]
    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None

    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        for col in range(BOARD_COLS - ancho_p + 1):
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero, lineas)   # <-- se pasa lineas
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot
######## Mejor placemente rapida
    def mejor_placement_rapido(matriz, pieza):
    """
    Estrategia rápida pero precisa para niveles altos (>=7).
    Evalúa todas las rotaciones, pero solo las 4 mejores columnas por rotación
    según una heurística simple (altura máxima en la zona + distancia al centro).
    Usa la misma función de puntuación que la estrategia robusta.
    """
    if pieza not in FORMAS_PIEZAS:
        return float('-inf'), None, None

    rotaciones = FORMAS_PIEZAS[pieza]
    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None
    
    # Alturas actuales para la heurística de filtrado
    alturas_actuales = alturas_columna(matriz)
    centro_ideal = 4.5  # centro del tablero (entre col 4 y 5)

    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        columnas_posibles = list(range(BOARD_COLS - ancho_p + 1))
        
        # Calcular un puntaje rápido para cada columna (menor es mejor)
        puntajes_columna = []
        for col in columnas_posibles:
            # Altura máxima en las columnas que ocupará la pieza (estimación)
            alturas_afectadas = [alturas_actuales[col + j] for j in range(ancho_p)]
            max_altura = max(alturas_afectadas) if alturas_afectadas else 0
            # Distancia al centro
            distancia_centro = abs((col + ancho_p / 2) - centro_ideal)
            # Heurística: priorizar columnas bajas y cercanas al centro
            punt_heuristica = max_altura + distancia_centro * 0.5
            puntajes_columna.append((col, punt_heuristica))
        
        # Ordenar y quedarse con las 4 mejores
        columnas_a_evaluar = [col for col, _ in sorted(puntajes_columna, key=lambda x: x[1])[:4]]
        
        for col in columnas_a_evaluar:
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero, lineas)   # <-- se pasa lineas
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot 
#######Ejecutar bot
def ejecutar_bot():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado (modo Tetris con énfasis en limpiar el tablero)")
    print("Niveles 1-6: Estrategia robusta")
    print("Niveles 7+: Estrategia rápida")
    print("Presiona Ctrl+C para detener")

    # Variables de estado
    pieza_actual = None
    pieza_en_hold = None
    ultimo_hold_usado = False
    contador_fallos = 0
    lineas_totales = 0
    nivel = 1  

    def calcular_nivel(lineas):
        if lineas < 3:
            return 1
        else:
            nivel = 1
            acum = 0
            meta = 3
            while lineas >= acum + meta:
                acum += meta
                nivel += 1
                meta += 2
            return nivel

    try:
        while True:
            # Capturar todas las zonas
            t = config["tablero"]
            s = config["siguientes"]
            h = config["hold"]

            # Calcular zona spawn
            cell_w, cell_h = obtener_tamano_celda(t)
            spawn_region = obtener_region_spawn(t)

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

            # Si hay pieza en spawn y no hay pieza actual, es nueva pieza
            if piezas_spawn and pieza_actual is None:
                pieza_actual = piezas_spawn[0][4]
                
                # Determinar si usamos estrategia rápida según el nivel
                nivel = calcular_nivel(lineas_totales)
                usar_estrategia_rapida = nivel >= 7

                # Calcular columna actual de la pieza en spawn
                x_spawn = piezas_spawn[0][0]
                columna_spawn = int(round(x_spawn / cell_w))
                columna_spawn = max(0, min(columna_spawn, BOARD_COLS-1))

                # Obtener próximas piezas (opcional, para información)
                proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []

                # Evaluar la mejor jugada para la pieza actual
                if usar_estrategia_rapida:
                    punt, col_objetivo, rot_objetivo = mejor_placement_rapido(matriz_tablero, pieza_actual)
                else:
                    punt, col_objetivo, rot_objetivo = mejor_placement(matriz_tablero, pieza_actual)

                # Opción de hold (simplificada, solo si la actual no se puede colocar)
                puede_colocar = col_objetivo is not None

                if puede_colocar:
                    # Colocar la pieza
                    _, lineas_eliminadas = simular_placement(matriz_tablero, pieza_actual, rot_objetivo, col_objetivo)
                    lineas_totales += lineas_eliminadas
                    exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                    keyboard, spawn_region, cell_w, nivel, usar_estrategia_rapida)
                    if exito:
                        ultimo_hold_usado = False
                    else:
                        contador_fallos += 1
                    pieza_actual = None
                else:
                    # Intentar hold si es posible
                    if not ultimo_hold_usado:
                        keyboard.press(Key.shift)
                        time.sleep(0.05 if not usar_estrategia_rapida else 0.03)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        pieza_actual = None
                        time.sleep(0.3 if not usar_estrategia_rapida else 0.15)
                    else:
                        # No se puede hacer nada, esperar
                        pieza_actual = None

            # Pequeña pausa entre iteraciones
            time.sleep(0.05 if nivel < 7 else 0.02)

    except KeyboardInterrupt:
        print("\nBot detenido")
    except Exception as e:
        print(f"Error en el bot: {e}")
        import traceback
        traceback.print_exc()
