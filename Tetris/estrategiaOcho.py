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
