def mejor_placement_rapido(matriz, pieza, nivel=7):
    """
    Estrategia intermedia para niveles altos (>=7). Evalúa las mejores 'num_rot' rotaciones
    (según heurística de altura) con todas las columnas. Si no encuentra colocación,
    evalúa las rotaciones restantes.
    """
    if pieza not in FORMAS_PIEZAS:
        return float('-inf'), None, None

    rotaciones = FORMAS_PIEZAS[pieza]
    alturas_actuales = alturas_columna(matriz)
    
    # Determinar cuántas rotaciones evaluar según nivel
    if nivel >= 10:
        num_rot = 1
    elif nivel >= 8:
        num_rot = 2
    else:
        num_rot = 3
    
    # Calcular heurística rápida para cada rotación: altura máxima que alcanzaría la pieza
    # si se coloca en la columna que minimice esa altura (simulación rápida)
    puntajes_rotacion = []
    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        altura_p = len(forma)
        # Encontrar la columna que minimiza la altura final de la pieza (altura de caída + altura de la pieza)
        mejor_altura = BOARD_ROWS + 1
        for col in range(BOARD_COLS - ancho_p + 1):
            # Calcular fila de caída aproximada (no es exacta pero da una idea)
            fila_caida = BOARD_ROWS - altura_p
            for i in range(altura_p):
                for j in range(ancho_p):
                    if forma[i][j] == 1:
                        # Buscar la primera fila desde abajo donde hay bloque
                        for f in range(fila_caida, -1, -1):
                            if matriz[f][col + j] == 1:
                                fila_caida = min(fila_caida, f - i - 1)
                                break
            altura_final = BOARD_ROWS - fila_caida
            if altura_final < mejor_altura:
                mejor_altura = altura_final
        puntajes_rotacion.append((rot_idx, mejor_altura))
    
    # Ordenar rotaciones por menor altura (mejor)
    puntajes_rotacion.sort(key=lambda x: x[1])
    rotaciones_a_evaluar = [r[0] for r in puntajes_rotacion[:num_rot]]
    
    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None
    
    # Evaluar rotaciones seleccionadas con todas las columnas
    for rot_idx in rotaciones_a_evaluar:
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
    
    # Si no se encontró colocación en las rotaciones seleccionadas, evaluar el resto
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

      # Y en ejecutar bot
  if usar_estrategia_rapida:
    punt, col_objetivo, rot_objetivo = mejor_placement_rapido(matriz_tablero, pieza_actual, nivel)
else:
    punt, col_objetivo, rot_objetivo = mejor_placement(matriz_tablero, pieza_actual)
    
    return mejor_punt, mejor_col, mejor_rot
