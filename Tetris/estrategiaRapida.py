def mejor_placement_rapido(matriz, pieza, num_candidatos=6):
    """
    Estrategia rápida pero precisa para niveles altos (>=7).
    Evalúa todas las rotaciones, pero solo las mejores 'num_candidatos' columnas por rotación
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
        
        # Si hay pocas columnas, evaluamos todas directamente
        if len(columnas_posibles) <= num_candidatos:
            columnas_a_evaluar = columnas_posibles
        else:
            # Calcular puntaje heurístico para cada columna (menor es mejor)
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
            
            # Ordenar y quedarse con las mejores 'num_candidatos'
            columnas_a_evaluar = [col for col, _ in sorted(puntajes_columna, key=lambda x: x[1])[:num_candidatos]]
        
        for col in columnas_a_evaluar:
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero, lineas)
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot
