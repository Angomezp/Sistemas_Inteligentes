def ejecutar_bot():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado (modo Tetris inteligente con estrategia adaptativa)")
    print("Niveles 1-6: Estrategia robusta")
    print("Niveles 7+: Estrategia rápida")
    print("Presiona Ctrl+C para detener")

    # Variables de estado
    pieza_actual = None
    pieza_en_hold = None
    ultimo_hold_usado = False
    contador_fallos = 0
    lineas_totales = 0  # acumulador de líneas eliminadas desde el inicio de la partida
    nivel = 1  

    # Función para calcular el nivel actual según las líneas totales
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

    # --- NUEVA FUNCIÓN DE SIMULACIÓN A DOS MOVIMIENTOS ---
    def simular_dos_movimientos(matriz, pieza_actual, pieza_hold, proximas, usar_estrategia_rapida):
        """
        Evalúa dos posibles caminos:
        1. Colocar pieza_actual ahora, luego colocar la primera de proximas (si existe)
        2. Hacer hold: intercambiar pieza_actual con pieza_hold (o guardar si no hay), luego colocar la nueva pieza actual,
           y luego la siguiente (si existe)
        Devuelve (accion, puntuacion_final, col_objetivo, rot_objetivo) para el primer movimiento.
        accion puede ser 'colocar', 'guardar', 'swap'
        """
        mejor_accion = None
        mejor_punt = float('-inf')
        mejor_col = None
        mejor_rot = None

        # Opción 1: colocar pieza actual ahora
        if usar_estrategia_rapida:
            punt1, col1, rot1 = mejor_placement_rapido(matriz, pieza_actual)
        else:
            punt1, col1, rot1 = mejor_placement(matriz, pieza_actual)

        if col1 is not None:
            nueva_matriz, lineas1 = simular_placement(matriz, pieza_actual, rot1, col1)
            if max(alturas_columna(nueva_matriz)) == 0:
                return 'colocar', 100000, col1, rot1

            if proximas:
                sig = proximas[0]
                if usar_estrategia_rapida:
                    punt2, col2, rot2 = mejor_placement_rapido(nueva_matriz, sig)
                else:
                    punt2, col2, rot2 = mejor_placement(nueva_matriz, sig)
                if col2 is not None:
                    matriz_final, lineas2 = simular_placement(nueva_matriz, sig, rot2, col2)
                    punt_final = puntuar_tablero(matriz_final)
                else:
                    punt_final = puntuar_tablero(nueva_matriz)
            else:
                punt_final = puntuar_tablero(nueva_matriz)

            if punt_final > mejor_punt:
                mejor_punt = punt_final
                mejor_accion = 'colocar'
                mejor_col = col1
                mejor_rot = rot1

        # Opción 2: hacer hold
        if pieza_hold is None:
            # Guardar pieza actual, la nueva actual será la primera de próximas
            if proximas:
                nueva_actual = proximas[0]
                resto_proximas = proximas[1:]
                if usar_estrategia_rapida:
                    punt_hold, col_hold, rot_hold = mejor_placement_rapido(matriz, nueva_actual)
                else:
                    punt_hold, col_hold, rot_hold = mejor_placement(matriz, nueva_actual)
                if col_hold is not None:
                    nueva_matriz_hold, lineas_h = simular_placement(matriz, nueva_actual, rot_hold, col_hold)
                    if max(alturas_columna(nueva_matriz_hold)) == 0:
                        return 'guardar', 100000, None, None
                    if resto_proximas:
                        sig2 = resto_proximas[0]
                        if usar_estrategia_rapida:
                            punt3, col3, rot3 = mejor_placement_rapido(nueva_matriz_hold, sig2)
                        else:
                            punt3, col3, rot3 = mejor_placement(nueva_matriz_hold, sig2)
                        if col3 is not None:
                            matriz_final_h, lineas_h2 = simular_placement(nueva_matriz_hold, sig2, rot3, col3)
                            punt_final_h = puntuar_tablero(matriz_final_h)
                        else:
                            punt_final_h = puntuar_tablero(nueva_matriz_hold)
                    else:
                        punt_final_h = puntuar_tablero(nueva_matriz_hold)

                    if punt_final_h > mejor_punt:
                        mejor_punt = punt_final_h
                        mejor_accion = 'guardar'
        else:
            # Hay pieza en hold, se puede hacer swap
            nueva_actual_swap = pieza_hold
            if usar_estrategia_rapida:
                punt_swap, col_swap, rot_swap = mejor_placement_rapido(matriz, nueva_actual_swap)
            else:
                punt_swap, col_swap, rot_swap = mejor_placement(matriz, nueva_actual_swap)
            if col_swap is not None:
                nueva_matriz_swap, lineas_sw = simular_placement(matriz, nueva_actual_swap, rot_swap, col_swap)
                if max(alturas_columna(nueva_matriz_swap)) == 0:
                    return 'swap', 100000, None, None
                if proximas:
                    sig_swap = proximas[0]
                    if usar_estrategia_rapida:
                        punt4, col4, rot4 = mejor_placement_rapido(nueva_matriz_swap, sig_swap)
                    else:
                        punt4, col4, rot4 = mejor_placement(nueva_matriz_swap, sig_swap)
                    if col4 is not None:
                        matriz_final_sw, lineas_sw2 = simular_placement(nueva_matriz_swap, sig_swap, rot4, col4)
                        punt_final_sw = puntuar_tablero(matriz_final_sw)
                    else:
                        punt_final_sw = puntuar_tablero(nueva_matriz_swap)
                else:
                    punt_final_sw = puntuar_tablero(nueva_matriz_swap)

                if punt_final_sw > mejor_punt:
                    mejor_punt = punt_final_sw
                    mejor_accion = 'swap'

        return mejor_accion, mejor_punt, mejor_col, mejor_rot

    # ----------------------------------------------------

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

                # Obtener próximas piezas (hasta 3)
                proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []

                # --- NUEVA EVALUACIÓN CON DOS PASOS ---
                accion, punt, col_objetivo, rot_objetivo = simular_dos_movimientos(
                    matriz_tablero, pieza_actual, pieza_en_hold, proximas_piezas, usar_estrategia_rapida)

                if accion == 'colocar' and col_objetivo is not None:
                    # Colocar la pieza actual
                    _, lineas_eliminadas = simular_placement(matriz_tablero, pieza_actual, rot_objetivo, col_objetivo)
                    lineas_totales += lineas_eliminadas
                    exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                    keyboard, spawn_region, cell_w, nivel, usar_estrategia_rapida)
                    if exito:
                        ultimo_hold_usado = False
                    else:
                        contador_fallos += 1
                    pieza_actual = None

                elif accion in ('guardar', 'swap'):
                    # Ejecutar hold
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
