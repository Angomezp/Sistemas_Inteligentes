def mejor_placement_rapido(matriz, pieza, nivel=7):
    """
    Estrategia para niveles >=7.
    - Nivel 7: evalúa las 3 mejores rotaciones y todas sus columnas.
    - Nivel >=8: evalúa la mejor rotación y solo las N mejores columnas,
                 seleccionadas por altura mínima (sin preferencia central).
    """
    if pieza not in FORMAS_PIEZAS:
        return float('-inf'), None, None

    rotaciones = FORMAS_PIEZAS[pieza]
    alturas_actuales = alturas_columna(matriz)

    # Determinar número de rotaciones y columnas según nivel
    if nivel >= 10:
        num_rot = 1
        num_cols = 3
    elif nivel >= 8:
        num_rot = 1
        num_cols = 4
    else:  # nivel 7
        num_rot = 3
        num_cols = BOARD_COLS  # todas

    # Calcular heurística de altura para cada rotación (menor altura mejor)
    puntajes_rotacion = []
    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        altura_p = len(forma)
        mejor_altura = BOARD_ROWS + 1
        for col in range(BOARD_COLS - ancho_p + 1):
            fila_caida = BOARD_ROWS - altura_p
            for i in range(altura_p):
                for j in range(ancho_p):
                    if forma[i][j] == 1:
                        for f in range(fila_caida, -1, -1):
                            if matriz[f][col + j] == 1:
                                fila_caida = min(fila_caida, f - i - 1)
                                break
            altura_final = BOARD_ROWS - fila_caida
            if altura_final < mejor_altura:
                mejor_altura = altura_final
        puntajes_rotacion.append((rot_idx, mejor_altura))

    puntajes_rotacion.sort(key=lambda x: x[1])
    rotaciones_a_evaluar = [r[0] for r in puntajes_rotacion[:num_rot]]

    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None

    for rot_idx in rotaciones_a_evaluar:
        forma = rotaciones[rot_idx]
        ancho_p = len(forma[0])
        columnas_posibles = list(range(BOARD_COLS - ancho_p + 1))

        # Para nivel 7, evaluar todas las columnas
        if nivel == 7:
            columnas_a_evaluar = columnas_posibles
        else:
            # Heurística para seleccionar las mejores columnas: solo la altura máxima
            # (menor altura, mejor; se ordena ascendente)
            puntajes_col = []
            for col in columnas_posibles:
                alturas_afectadas = [alturas_actuales[col + j] for j in range(ancho_p)]
                max_altura = max(alturas_afectadas) if alturas_afectadas else 0
                # Sin distancia al centro, solo priorizar la altura más baja
                puntajes_col.append((col, max_altura))
            # Ordenar por altura (menor mejor)
            puntajes_col.sort(key=lambda x: x[1])
            columnas_a_evaluar = [c[0] for c in puntajes_col[:num_cols]]

        for col in columnas_a_evaluar:
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero, lineas)
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    # Fallback: si no se encontró nada, evaluar rotaciones restantes (raro)
    if mejor_col is None:
        rotaciones_restantes = [i for i in range(len(rotaciones)) if i not in rotaciones_a_evaluar]
        for rot_idx in rotaciones_restantes:
            forma = rotaciones[rot_idx]
            ancho_p = len(forma[0])
            for col in range(BOARD_COLS - ancho_p + 1):
                nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
                if nuevo_tablero is None:
                    continue
                punt = puntuar_tablero(nuevo_tablero, lineas)
                if punt > mejor_punt:
                    mejor_punt = punt
                    mejor_col = col
                    mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot
