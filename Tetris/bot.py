########################################
# BOT
########################################

def ejecutar_bot():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado")
    print("Presiona Ctrl+C para detener")
    
    # Variables de estado
    pieza_actual = None
    pieza_guardada = None
    frame_skip = 0
    
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
            
            # Si hay pieza en spawn y no hay pieza actual, es nueva pieza
            if piezas_spawn and pieza_actual is None:
                pieza_actual = piezas_spawn[0][4]  # El tipo de pieza
                print(f"Nueva pieza detectada: {pieza_actual}")
                
                # Encontrar mejor posición
                mejor_col = encontrar_mejor_posicion(pieza_actual, matriz_tablero)
                
                if mejor_col is not None:
                    # Calcular desplazamiento necesario
                    col_actual = 4  # Asumimos que spawn en columna 4
                    desplazamiento = mejor_col - col_actual
                    
                    # Mover la pieza
                    print(f"Moviendo a columna {mejor_col}, desplazamiento: {desplazamiento}")
                    
                    # Mover horizontalmente
                    if desplazamiento > 0:
                        for _ in range(desplazamiento):
                            keyboard.press('right')
                            time.sleep(0.05)
                            keyboard.release('right')
                            time.sleep(0.05)
                    elif desplazamiento < 0:
                        for _ in range(abs(desplazamiento)):
                            keyboard.press('left')
                            time.sleep(0.05)
                            keyboard.release('left')
                            time.sleep(0.05)
                    
                    # Bajar la pieza inmediatamente
                    time.sleep(0.1)
                    keyboard.press('space')
                    time.sleep(0.05)
                    keyboard.release('space')
                    
                    pieza_actual = None  # Resetear para siguiente pieza
                else:
                    # No hay espacio, intentar guardar
                    if piezas_hold:
                        # Hay pieza en hold, hacer swap
                        print("No hay espacio, haciendo swap con hold")
                        keyboard.press('shift')
                        time.sleep(0.05)
                        keyboard.release('shift')
                        
                        # La pieza que estaba en hold ahora es la actual
                        if piezas_hold:
                            pieza_actual = piezas_hold[0][4]
                            
                            # Esperar a que aparezca la pieza del hold
                            time.sleep(0.3)
                            
                            # Mover la pieza del hold
                            mejor_col = encontrar_mejor_posicion(pieza_actual, matriz_tablero)
                            if mejor_col is not None:
                                col_actual = 4
                                desplazamiento = mejor_col - col_actual
                                
                                if desplazamiento > 0:
                                    for _ in range(desplazamiento):
                                        keyboard.press('right')
                                        time.sleep(0.05)
                                        keyboard.release('right')
                                        time.sleep(0.05)
                                elif desplazamiento < 0:
                                    for _ in range(abs(desplazamiento)):
                                        keyboard.press('left')
                                        time.sleep(0.05)
                                        keyboard.release('left')
                                        time.sleep(0.05)
                                
                                time.sleep(0.1)
                                keyboard.press('space')
                                time.sleep(0.05)
                                keyboard.release('space')
                    else:
                        # No hay pieza en hold, guardar la actual
                        print("No hay espacio, guardando en hold")
                        keyboard.press('shift')
                        time.sleep(0.05)
                        keyboard.release('shift')
                    
                    pieza_actual = None
            
            # Pequeña pausa para no saturar
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nBot detenido")

def encontrar_mejor_posicion(pieza, matriz_tablero):
    """
    Encuentra la columna más cercana al centro donde la pieza pueda caber
    Retorna la columna o None si no hay espacio
    """
    # Definir formas de las piezas (matrices 4x4)
    formas = {
        'I': [[1,1,1,1]],
        'O': [[1,1],[1,1]],
        'T': [[0,1,0],[1,1,1]],
        'S': [[0,1,1],[1,1,0]],
        'Z': [[1,1,0],[0,1,1]],
        'J': [[1,0,0],[1,1,1]],
        'L': [[0,0,1],[1,1,1]]
    }
    
    if pieza not in formas:
        return 4  # Default al centro
    
    forma = formas[pieza]
    altura_pieza = len(forma)
    ancho_pieza = len(forma[0])
    
    # Probar cada columna posible
    posiciones_validas = []
    
    for col in range(BOARD_COLS - ancho_pieza + 1):
        # Verificar si la pieza cabe en esta columna
        cabe = True
        
        for fila_tablero in range(BOARD_ROWS - altura_pieza + 1):
            # Verificar si esta posición está libre
            pos_libre = True
            for i in range(altura_pieza):
                for j in range(ancho_pieza):
                    if forma[i][j] == 1:
                        # Verificar si la celda está ocupada o fuera del tablero
                        if (fila_tablero + i >= BOARD_ROWS or 
                            col + j >= BOARD_COLS or
                            matriz_tablero[fila_tablero + i][col + j] == 1):
                            pos_libre = False
                            break
                if not pos_libre:
                    break
            
            if pos_libre:
                # Encontramos una posición válida
                posiciones_validas.append((col, fila_tablero))
                break
    
    if not posiciones_validas:
        return None
    
    # Encontrar la columna más cercana al centro (columna 4-5)
    centro = 4.5
    mejor_col = min(posiciones_validas, key=lambda x: abs(x[0] + 0.5 - centro))[0]
    
    return mejor_col
