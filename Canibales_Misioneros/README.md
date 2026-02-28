# Sistemas Inteligentes / Canibales y Misioneros
**Descripción**

Proyecto sencillo que automatiza el juego "Misioneros y Caníbales" en pantalla usando detección por plantillas y un solucionador BFS.

El script principal es [Canibales_Misioneros/bot_AzucarMorena.py](Canibales_Misioneros/bot_AzucarMorena.py) y utiliza imágenes de referencia en [Canibales_Misioneros/sprites](Canibales_Misioneros/sprites) para detectar los elementos en la pantalla.

El link de juego es el siguiente: (https://www.novelgames.com/es/missionaries/) 

**Estructura del repositorio**

- [Canibales_Misioneros/bot_AzucarMorena.py](Canibales_Misioneros/bot_AzucarMorena.py): código del bot, detección por plantilla, lógica de control y solucionador BFS.
- [Canibales_Misioneros/sprites](Canibales_Misioneros/sprites): plantillas de imagen usadas para la detección (misionero, caníbal, balsa).

**Requisitos**

Instalar las dependencias de Python (recomendado dentro de un virtualenv):

```
pip install opencv-python numpy pyautogui pillow pynput
```

Nota: `tkinter` viene incluido en la mayoría de las instalaciones de Python en Windows.

**Cómo usar**

1. Abrir el juego en pantalla y asegurarse de que la zona del juego está visible.
2. Ejecutar:

	```
	python Canibales_Misioneros/bot_AzucarMorena.py
	```

	Si estas en el root del repositorio, en caso de que estes en la carpeta `Canibales_Misioneros` ejecuta:

	```
	python bot_AzucarMorena.py
	```


3. En el menú del script se puede:
	- Ejecutar (presionar `ALT+X` tras seleccionar la opción 1). El bot intentará resolver el puzzle automáticamente una vez se haya calibrado el area de deteccion del juego y se haya abierto con anterioridad el [juego](https://www.novelgames.com/es/missionaries/) y se vea en pantalla (El juego puede iniciar en cualquier estado valido y el bot le intentara dar solucion).
	- Probar detección: imprime el conteo detectado y ofrece visualizar la detección.
	- Visualizar: abre una ventana con la detección en tiempo real.
	- Calibrar: sirve para medir la zona de juego mediante posiciones del cursor.

**Cómo funciona (resumen técnico)**

- Detección: el bot captura una región fija de la pantalla (`GAME_X1..GAME_X2`, `GAME_Y1..GAME_Y2`) y busca coincidencias por plantilla con `cv2.matchTemplate`.
- Agrupa detecciones cercanas para evitar duplicados y clasifica las entidades (misioneros, caníbales, balsa) según la mitad del área y la proximidad a la balsa.
- Solución: calcula una secuencia de movimientos válida usando BFS sobre el estado (nº misioneros y caníbales en la orilla derecha y la posición de la balsa).
- Ejecución: realiza clicks mediante `pyautogui` para seleccionar personajes y mover la balsa; valida el estado tras cada movimiento.

**Ajustes y calibración**

- Los parámetros de detección y offsets están definidos en la parte superior de `bot_AzucarMorena.py`. Ajustar `UMBRAL`, `UMBRAL_BALSA`, offsets y las coordenadas del área (`GAME_X1..GAME_Y2`) según resolución y escala.
- Para calibrar manualmente el area de deteccion del juego usar la opción 4 del menú (calibrar) y seguir las instrucciones.

**Advertencia**

- El bot hace clics automáticos en la pantalla (riesgo de interacción no deseada). Cerrar el juego o mover la ventana puede desincronizar la detección.
