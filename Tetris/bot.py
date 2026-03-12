def colocar_pieza_mejorada(pieza, columna_spawn_inicial, columna_objetivo, rotacion_objetivo, keyboard, spawn_region, cell_w, nivel):
    """
    Coloca la pieza desde su posición actual de spawn hasta la posición objetivo.
    Los tiempos de espera se ajustan según el nivel (mayor nivel = menor espera).
    """
    # Definir tiempos base (en segundos) - estos son los que funcionan bien hasta nivel 4
    tiempos_base = {
        'pulsacion': 0.05,      # duración de la pulsación de tecla
        'post_pulsacion': 0.05, # espera después de soltar la tecla
        'post_rotacion': 0.1,   # espera después de cada rotación antes de siguiente acción
        'pre_soltar': 0.1,      # espera antes de soltar la pieza con espacio
        'reintento': 0.1        # espera entre reintentos de detección
    }

    # Factor de escala según el nivel (a mayor nivel, menor tiempo)
    if nivel <= 4:
        factor = 1.0
    elif nivel == 5:
        factor = 0.8
    elif nivel == 6:
        factor = 0.6
    elif nivel == 7:
        factor = 0.5
    else:
        factor = 0.4  # niveles muy altos

    # Calcular tiempos reales (con un mínimo para evitar valores demasiado pequeños)
    t_puls = max(tiempos_base['pulsacion'] * factor, 0.02)
    t_post_puls = max(tiempos_base['post_pulsacion'] * factor, 0.02)
    t_post_rot = max(tiempos_base['post_rotacion'] * factor, 0.03)
    t_pre_soltar = max(tiempos_base['pre_soltar'] * factor, 0.05)
    t_reintento = max(tiempos_base['reintento'] * factor, 0.03)

    num_rot = len(FORMAS_PIEZAS[pieza])
    rot_inicial = 0
    rot_necesarias = (rotacion_objetivo - rot_inicial) % num_rot

    print(f"Rotando {rot_necesarias} veces (nivel {nivel}, factor {factor:.2f})")

    # Aplicar rotaciones una por una
    for _ in range(rot_necesarias):
        keyboard.press(Key.up)
        time.sleep(t_puls)
        keyboard.release(Key.up)
        time.sleep(t_post_rot)  # espera después de la rotación

    # Re-detectar la pieza después de rotar (con reintento)
    columna_actual = columna_spawn_inicial
    for intento in range(2):
        time.sleep(t_reintento)
        spawn_img = capture_screen(spawn_region)
        piezas_detectadas = detectar_piezas_spawn(spawn_img)
        if piezas_detectadas:
            x_spawn = piezas_detectadas[0][0]
            columna_actual = int(round(x_spawn / cell_w))
            columna_actual = max(0, min(columna_actual, BOARD_COLS-1))
            print(f"Posición después de rotar (intento {intento+1}): columna {columna_actual}")
            break
        else:
            print(f"Intento {intento+1} fallido al detectar pieza después de rotar")
    else:
        print("No se pudo detectar la pieza después de rotar. Usando columna inicial.")

    desplazamiento = columna_objetivo - columna_actual
    print(f"Moviendo desde col {columna_actual} a col {columna_objetivo} (desp={desplazamiento})")

    # Mover horizontalmente
    if desplazamiento > 0:
        for _ in range(desplazamiento):
            keyboard.press(Key.right)
            time.sleep(t_puls)
            keyboard.release(Key.right)
            time.sleep(t_post_puls)
    elif desplazamiento < 0:
        for _ in range(abs(desplazamiento)):
            keyboard.press(Key.left)
            time.sleep(t_puls)
            keyboard.release(Key.left)
            time.sleep(t_post_puls)

    # Soltar la pieza
    print("Soltando pieza...")
    time.sleep(t_pre_soltar)
    keyboard.press(Key.space)
    time.sleep(t_puls)
    keyboard.release(Key.space)

    return True

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
    pieza_guardada_hold = None
    ultimo_hold_usado = False
    contador_fallos = 0
    lineas_totales = 0  # acumulador de líneas eliminadas desde el inicio de la partida

    # Función para calcular el nivel actual según las líneas totales
    def calcular_nivel(lineas):
        if lineas < 3:
            return 1
        else:
            # Buscar el mayor n tal que n*(n+2) <= lineas
            n = 1
            while n*(n+2) <= lineas:
                n += 1
            return n  # el nivel es n (porque cuando se alcanzan las líneas del nivel n, se pasa al n+1? Cuidado)
            # Si lineas = 3, n=1 da 3 <=3, luego n=2 da 8>3, entonces n=2? No, queremos nivel 2 cuando se han completado 3 líneas (pasaste a nivel 2). Entonces nivel = n donde n*(n+2) <= lineas? Probemos:
            # lineas=2 -> n=1 da 3>2, entonces n=1 no cumple, así que nivel=1.
            # lineas=3 -> n=1 cumple, n=2 no, entonces n=1, pero debería ser nivel 2. Así que nivel = n+1 cuando se alcanza la meta.
            # Mejor: nivel = 1 + el mayor n tal que n*(n+2) < lineas? 
            # Definamos: meta para nivel k es la suma hasta k, es decir, nivel k requiere haber completado al menos la suma de los primeros k-1 términos? 
            # Normalmente: nivel 1: 0-2 líneas (aún no se ha pasado), nivel 2: después de 3 líneas, etc.
            # Entonces si lineas >= 3, estás en nivel 2. Si lineas >= 8, nivel 3, etc.
            # Entonces nivel = 1 + el mayor n tal que n*(n+2) <= lineas? Para lineas=3, n=1 da 3<=3, entonces n=1, nivel=2. Para lineas=8, n=2 da 8<=8, nivel=3. Correcto.
            # Para lineas=2, ningún n cumple porque 1*3=3>2, entonces nivel=1.
            # Así que:
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
                    # Antes de colocar, simulamos para conocer las líneas que se eliminarán
                    _, lineas_eliminadas = simular_placement(matriz_tablero, pieza_actual, rot_objetivo, col_objetivo)
                    lineas_totales += lineas_eliminadas
                    # Calcular nivel actual
                    nivel = calcular_nivel(lineas_totales)
                    print(f"Líneas totales: {lineas_totales} - Nivel: {nivel}")

                    print(f"Colocando {pieza_actual} (puntuación: {punt_actual:.1f})")
                    exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                    keyboard, spawn_region, cell_w, nivel)
                    if exito:
                        print(f"Pieza {pieza_actual} colocada")
                        ultimo_hold_usado = False
                    else:
                        print(f"Error colocando {pieza_actual}")
                        contador_fallos += 1
                    pieza_actual = None

                elif punt_hold > float('-inf') and punt_hold > punt_actual:
                    # Acción de hold: no se colocan piezas, por lo tanto no hay líneas nuevas
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
                    # No se puede colocar ni usar hold
                    print(f"No se puede colocar {pieza_actual} y hold no disponible o no mejora")
                    if puede_colocar_actual:
                        # Forzar colocación
                        _, lineas_eliminadas = simular_placement(matriz_tablero, pieza_actual, rot_objetivo, col_objetivo)
                        lineas_totales += lineas_eliminadas
                        nivel = calcular_nivel(lineas_totales)
                        print(f"Líneas totales: {lineas_totales} - Nivel: {nivel}")
                        exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                        keyboard, spawn_region, cell_w, nivel)
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
