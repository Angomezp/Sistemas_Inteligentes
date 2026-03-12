def colocar_pieza_mejorada(pieza, columna_spawn_inicial, columna_objetivo, rotacion_objetivo, keyboard, spawn_region, cell_w, nivel):
    """
    Coloca la pieza desde su posición actual de spawn hasta la posición objetivo.
    Los tiempos de espera se ajustan según el nivel.
    Las rotaciones se realizan con las teclas óptimas: X (horario), Z (antihorario), A (180°).
    """
    # Definir tiempos base (en segundos) - estos funcionan bien hasta nivel 4
    tiempos_base = {
        'pulsacion': 0.03,      # duración de la pulsación de tecla
        'post_pulsacion': 0.05, # espera después de soltar la tecla
        'post_rotacion': 0.1,   # espera después de cada rotación
        'pre_soltar': 0.07,      # espera antes de soltar la pieza
        'reintento': 0.1        # espera entre reintentos de detección
    }

    # Factor de escala según el nivel (a mayor nivel, menor tiempo)
    if nivel <= 4:
        factor = 1.0
    elif nivel == 5:
        factor = 0.7
    elif nivel == 6:
        factor = 0.6
    elif nivel == 7:
        factor = 0.5
    else:
        factor = 0.4  # niveles muy altos

    # Calcular tiempos reales con mínimo
    t_puls = max(tiempos_base['pulsacion'] * factor, 0.02)
    t_post_puls = max(tiempos_base['post_pulsacion'] * factor, 0.02)
    t_post_rot = max(tiempos_base['post_rotacion'] * factor, 0.03)
    t_pre_soltar = max(tiempos_base['pre_soltar'] * factor, 0.05)
    t_reintento = max(tiempos_base['reintento'] * factor, 0.03)

    num_rot = len(FORMAS_PIEZAS[pieza])

    # --- Rotación óptima según la pieza y el objetivo ---
    if num_rot == 4:
        if rotacion_objetivo == 1:
            tecla = 'x'
            pulsaciones = 1
        elif rotacion_objetivo == 2:
            tecla = 'a'
            pulsaciones = 1
        elif rotacion_objetivo == 3:
            tecla = 'z'
            pulsaciones = 1
        else:
            pulsaciones = 0
    elif num_rot == 2:
        if rotacion_objetivo == 1:
            tecla = 'x'   # también podría ser 'z', ambas sirven
            pulsaciones = 1
        else:
            pulsaciones = 0
    else:  # num_rot == 1 (pieza O)
        pulsaciones = 0

    if pulsaciones > 0:
        print(f"Aplicando rotación: tecla '{tecla}' (para objetivo {rotacion_objetivo})")
        keyboard.press(tecla)
        time.sleep(t_puls)
        keyboard.release(tecla)
        time.sleep(t_post_rot)
    else:
        print("No se requiere rotación")

    # --- Re‑detectar la pieza después de rotar (con reintento) ---
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
