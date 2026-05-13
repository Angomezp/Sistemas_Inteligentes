# Tetris - Bot Automático

**Grupo:** Azúcar Morena  
**Integrantes:**
- [Leidy Yulliana Quiñones González](https://github.com/LeidyQG)
- [Ángel David Gómez Pastrana](https://github.com/Angomezp)

**Materia:** Sistemas Inteligentes

## Descripción
Bot automático que juega Tetris analizando el tablero en tiempo real y calculando movimientos óptimos mediante visión por computadora.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| **bot_AzucarMorena.py** | Bot principal que captura pantalla, analiza el tablero y ejecuta movimientos automáticos |
| **tetrio_config.json** | Configuración del bot (coordenadas, tolerancia de color, parámetros del tablero) |
| **config.ttc** | Archivo de configuración del juego Tetris |
| **botSave.txt** | Registro/log de ejecuciones del bot |
| **sprites/** | Recursos gráficos |

## Características
- Detección de piezas en tiempo real mediante captura de pantalla
- Análisis del tablero para identificar posiciones óptimas
- Ejecución automática de movimientos y rotaciones
- Calibración configurable de coordenadas y colores

## Uso
```bash
python bot_AzucarMorena.py
```
