def puntuar_tablero(matriz):
    """
    Evalúa la bondad del tablero. Mayor puntuación es mejor.
    Factores: menos huecos, menos altura, menos irregularidad.
    Si el tablero está completamente vacío (altura máxima 0), se da una recompensa enorme.
    """
    alturas = alturas_columna(matriz)
    huecos = contar_huecos(matriz)
    bump = calcular_bumpiness(alturas)
    altura_max = max(alturas) if alturas else 0

    # Recompensa masiva si el tablero queda limpio
    if altura_max == 0:
        return 10000

    PESO_HUECOS = -25
    PESO_ALTURA = -10
    PESO_BUMP = -2

    puntuacion = (PESO_HUECOS * huecos +
                  PESO_ALTURA * altura_max +
                  PESO_BUMP * bump)
    return puntuacion
