#Zona rosada, malla y deteccion de letras agregada

import os
import cv2
import numpy as np
import mss
import json
from pynput.keyboard import Controller
import time

keyboard = Controller()

CONFIG_FILE = "tetrio_config.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, CONFIG_FILE)

########################################
# TOLERANCIA DE COLOR
########################################

COLOR_TOLERANCE = 30

########################################
# CONFIGURACION TABLERO
########################################

BOARD_ROWS = 20
BOARD_COLS = 10
SPAWN_ROWS = 4

########################################
# COLORES DE PIEZAS POR ZONA
########################################

# Colores para zona spawn (arriba del tablero)
SPAWN_COLORS = {
    "I": np.array([255,255,0]),
    "Z": np.array([75,68,191]),
    "J": np.array([255,0,0]),
    "L": np.array([66,115,196]),
    "O": np.array([84,178,200]),
    "T": np.array([163,64,173]),
    "S": np.array([55,255,175])
}

# Colores para zona next/hold
NEXT_HOLD_COLORS = {
    "I": np.array([255,255,0]),
    "Z": np.array([75,68,191]),
    "J": np.array([255,0,0]),
    "L": np.array([66,115,196]),
    "O": np.array([84,178,200]),
    "T": np.array([163,64,173]),
    "S": np.array([55,255,175])
}

########################################
# CAPTURA DE PANTALLA
########################################

def capture_screen(region=None):
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]
        else:
            monitor = {
                "top": region[1],
                "left": region[0],
                "width": region[2],
                "height": region[3]
            }

        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

drawing = False
ix, iy = -1, -1
rect = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()
            cv2.rectangle(img_copy,(ix,iy),(x,y),(0,255,0),2)
            cv2.imshow("calibracion",img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix,iy,x-ix,y-iy)
        cv2.rectangle(param,(ix,iy),(x,y),(0,255,0),2)
        cv2.imshow("calibracion",param)

def calibrar_area(nombre):
    global rect

    time.sleep(0.5)
    screen = capture_screen()
    rect = None

    cv2.namedWindow("calibracion")
    cv2.setMouseCallback("calibracion",draw_rectangle,screen)

    print(f"\nSelecciona el area de {nombre}")
    print("Arrastra con el mouse y presiona 's'")

    while True:
        cv2.imshow("calibracion",screen)
        key = cv2.waitKey(1)
        if key == ord('s') and rect is not None:
            break

    cv2.destroyAllWindows()
    return rect

def modo_calibracion():
    config = {}
    config["tablero"] = calibrar_area("TABLERO")
    config["siguientes"] = calibrar_area("SIGUIENTES")
    config["hold"] = calibrar_area("HOLD")

    with open(CONFIG_PATH,"w") as f:
        json.dump(config,f)

    print("Calibración guardada")

########################################
# CALIBRACIÓN DE COLORES - SPAWN
########################################

def calibrar_colores_spawn():
    print("\nCalibración de colores - ZONA SPAWN")
    print("Haz click sobre cada pieza cuando se te indique (asegúrate de que esté en la zona de spawn)")
    
    # Usar la zona de spawn para capturar
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    # Calcular zona spawn
    t = config["tablero"]
    cell_w, cell_h = obtener_tamano_celda(t)
    spawn_height = int(cell_h * SPAWN_ROWS)
    
    spawn_region = (
        t[0],
        t[1] - spawn_height,
        t[2],
        spawn_height
    )
    
    # Capturar imagen de la zona spawn
    screen = capture_screen(spawn_region)
    
    piezas_orden = ["I","J","L","O","S","T","Z"]
    colores = {}
    indice = [0]

    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pieza_actual = piezas_orden[indice[0]]
            color = screen[y,x]
            print(f"Color capturado para {pieza_actual} (spawn): {color}")
            colores[pieza_actual] = color.tolist()
            indice[0] += 1

            if indice[0] < len(piezas_orden):
                print(f"Haz click sobre la pieza {piezas_orden[indice[0]]} en zona spawn")
            else:
                print("Todas las piezas de spawn capturadas")

    cv2.namedWindow("calibrar_colores_spawn")
    cv2.setMouseCallback("calibrar_colores_spawn",click)

    print(f"Haz click sobre la pieza {piezas_orden[0]} en zona spawn")

    while True:
        cv2.imshow("calibrar_colores_spawn",screen)
        if indice[0] >= len(piezas_orden):
            break
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Guardar configuración
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    else:
        config = {}

    if "colores" not in config:
        config["colores"] = {}
    
    config["colores"]["spawn"] = colores
    config["tolerancia_color"] = COLOR_TOLERANCE

    with open(CONFIG_PATH,"w") as f:
        json.dump(config,f, indent=4)

    print("Colores de spawn calibrados y guardados")

########################################
# CALIBRACIÓN DE COLORES - NEXT/HOLD
########################################

def calibrar_colores_next_hold():
    print("\nCalibración de colores - ZONA NEXT/HOLD")
    print("Haz click sobre cada pieza cuando se te indique (puedes usar la zona de next o hold)")
    
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    
    # Usar la zona de siguientes para capturar (asumiendo que ahí se ven mejor)
    screen = capture_screen(config["siguientes"])
    
    piezas_orden = ["I","J","L","O","S","T","Z"]
    colores = {}
    indice = [0]

    def click(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pieza_actual = piezas_orden[indice[0]]
            color = screen[y,x]
            print(f"Color capturado para {pieza_actual} (next/hold): {color}")
            colores[pieza_actual] = color.tolist()
            indice[0] += 1

            if indice[0] < len(piezas_orden):
                print(f"Haz click sobre la pieza {piezas_orden[indice[0]]} en zona next/hold")
            else:
                print("Todas las piezas de next/hold capturadas")

    cv2.namedWindow("calibrar_colores_next_hold")
    cv2.setMouseCallback("calibrar_colores_next_hold",click)

    print(f"Haz click sobre la pieza {piezas_orden[0]} en zona next/hold")

    while True:
        cv2.imshow("calibrar_colores_next_hold",screen)
        if indice[0] >= len(piezas_orden):
            break
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Guardar configuración
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    else:
        config = {}

    if "colores" not in config:
        config["colores"] = {}
    
    config["colores"]["next_hold"] = colores
    config["tolerancia_color"] = COLOR_TOLERANCE

    with open(CONFIG_PATH,"w") as f:
        json.dump(config,f, indent=4)

    print("Colores de next/hold calibrados y guardados")

########################################
# CALCULAR TAMAÑO DE CELDA
########################################

def obtener_tamano_celda(region_tablero):
    width = region_tablero[2]
    height = region_tablero[3]
    cell_w = width / BOARD_COLS
    cell_h = height / BOARD_ROWS
    return cell_w, cell_h

########################################
# OBTENER ZONA SPAWN (3 FILAS ARRIBA)
########################################

def obtener_region_spawn(region_tablero):
    cell_w, cell_h = obtener_tamano_celda(region_tablero)
    spawn_height = int(cell_h * SPAWN_ROWS)
    spawn_region = (
        region_tablero[0],
        int(region_tablero[1] - spawn_height),
        region_tablero[2],
        spawn_height
    )
    return spawn_region

########################################
# MATRIZ TABLERO
########################################

def obtener_matriz_tablero(tablero_img, region_tablero):
    matriz = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
    gray = cv2.cvtColor(tablero_img, cv2.COLOR_BGR2GRAY)
    cell_w, cell_h = obtener_tamano_celda(region_tablero)

    for fila in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x1 = int(col * cell_w)
            y1 = int(fila * cell_h)
            x2 = int(x1 + cell_w)
            y2 = int(y1 + cell_h)

            if y2 > tablero_img.shape[0] or x2 > tablero_img.shape[1]:
                continue

            celda = gray[y1:y2, x1:x2]
            promedio = np.mean(celda)

            if promedio > 40:
                matriz[fila][col] = 1
            else:
                matriz[fila][col] = 0

    return matriz

########################################
# DIBUJAR MALLA
########################################

def dibujar_malla(screen, region):
    cell_w, cell_h = obtener_tamano_celda(region)
    x_offset = region[0]
    y_offset = region[1]

    for c in range(BOARD_COLS + 1):
        x = int(x_offset + c * cell_w)
        cv2.line(
            screen,
            (x, y_offset),
            (x, int(y_offset + BOARD_ROWS * cell_h)),
            (255,255,255),
            1
        )

    for r in range(1, BOARD_ROWS):
        y = int(y_offset + r * cell_h)
        cv2.line(
            screen,
            (x_offset, y),
            (int(x_offset + BOARD_COLS * cell_w), y),
            (255,255,255),
            1
        )

########################################
# DIBUJAR OCUPACION
########################################

def dibujar_ocupacion(screen, region, matriz):
    cell_w, cell_h = obtener_tamano_celda(region)
    x_offset = region[0]
    y_offset = region[1]

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            if matriz[r][c] == 1:
                cx = int(x_offset + c * cell_w + cell_w/2)
                cy = int(y_offset + r * cell_h + cell_h/2)
                cv2.circle(
                    screen,
                    (cx,cy),
                    4,
                    (0,255,255),
                    -1
                )

########################################
# DIBUJAR PIEZAS CORREGIDA
########################################

def dibujar_piezas(screen, region, piezas, color, offset_adicional=(0,0)):
    """
    Dibuja piezas en la pantalla
    region: tupla (x, y, w, h) de la zona donde se detectaron
    piezas: lista de (x, y, w, h, tipo) relativas a la imagen de la zona
    offset_adicional: para ajustar si la región no empieza en (0,0)
    """
    x_offset = region[0] + offset_adicional[0]
    y_offset = region[1] + offset_adicional[1]
    
    for (x, y, w, h, tipo) in piezas:
        # Centrar el texto en el bounding box
        px = x_offset + x + w//2 - 10
        py = y_offset + y + h//2 + 5
        
        cv2.putText(
            screen,
            tipo,
            (px, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

########################################
# CARGAR COLORES POR ZONA
########################################

def cargar_colores_calibrados():
    global SPAWN_COLORS, NEXT_HOLD_COLORS, COLOR_TOLERANCE

    if not os.path.exists(CONFIG_PATH):
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if "colores" in config:
        if "spawn" in config["colores"]:
            for pieza, color in config["colores"]["spawn"].items():
                SPAWN_COLORS[pieza] = np.array(color)
            print("Colores de spawn cargados")
        
        if "next_hold" in config["colores"]:
            for pieza, color in config["colores"]["next_hold"].items():
                NEXT_HOLD_COLORS[pieza] = np.array(color)
            print("Colores de next/hold cargados")

    if "tolerancia_color" in config:
        COLOR_TOLERANCE = config["tolerancia_color"]

########################################
# CLASIFICAR PIEZA PARA SPAWN
########################################

def clasificar_pieza_spawn(img, contorno, x, y, w, h):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    
    # Calcular color medio
    mean_color = cv2.mean(img, mask=mask)[:3]
    mean_color = np.array(mean_color)
    
    # Calcular ratio de aspecto
    ratio = w / h if h != 0 else 1
    
    # Detección especial para pieza I (muy alargada)
    if ratio > 2.0:
        dist_i = np.linalg.norm(mean_color - SPAWN_COLORS["I"])
        if dist_i < COLOR_TOLERANCE * 1.5:
            return "I"
    
    # Para piezas T, L, J, Z (las problemáticas)
    mejor = None
    mejor_dist = COLOR_TOLERANCE
    
    for pieza, color in SPAWN_COLORS.items():
        dist = np.linalg.norm(mean_color - color)
        
        # Dar un pequeño bonus a las piezas que coinciden por forma
        if pieza in ["T", "L", "J", "Z"]:
            if 0.7 < ratio < 1.8:
                dist *= 0.9
        
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = pieza
    
    return mejor

########################################
# CLASIFICAR PIEZA PARA NEXT/HOLD
########################################

def clasificar_pieza_next_hold(img, contorno, x, y, w, h):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)
    
    # Calcular color medio
    mean_color = cv2.mean(img, mask=mask)[:3]
    mean_color = np.array(mean_color)
    
    # Calcular ratio de aspecto
    ratio = w / h if h != 0 else 1
    
    # Detección especial para pieza I (muy alargada)
    if ratio > 2.0:
        dist_i = np.linalg.norm(mean_color - NEXT_HOLD_COLORS["I"])
        if dist_i < COLOR_TOLERANCE * 1.5:
            return "I"
    
    # Para piezas T, L, J, Z (las problemáticas)
    mejor = None
    mejor_dist = COLOR_TOLERANCE
    
    for pieza, color in NEXT_HOLD_COLORS.items():
        dist = np.linalg.norm(mean_color - color)
        
        # Dar un pequeño bonus a las piezas que coinciden por forma
        if pieza in ["T", "L", "J", "Z"]:
            if 0.7 < ratio < 1.8:
                dist *= 0.9
        
        if dist < mejor_dist:
            mejor_dist = dist
            mejor = pieza
    
    return mejor

########################################
# DETECTAR PIEZAS EN ZONA SPAWN
########################################

def detectar_piezas_spawn(img):
    """
    Versión para spawn que usa SPAWN_COLORS
    """
    if img.size == 0:
        return []
    
    # Convertir a HSV para mejor detección de color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Crear máscara de saturación (los colores tienen alta saturación)
    saturation = hsv[:, :, 1]
    
    # Umbral adaptativo para separar piezas del fondo negro
    _, th_sat = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
    
    # Operaciones morfológicas para limpiar
    kernel = np.ones((3, 3), np.uint8)
    th_sat = cv2.morphologyEx(th_sat, cv2.MORPH_OPEN, kernel)
    th_sat = cv2.morphologyEx(th_sat, cv2.MORPH_CLOSE, kernel)
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(th_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    piezas = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        
        # Filtrar por área
        if area < 50:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        # Verificar que el bounding box sea razonable
        if w < 5 or h < 5:
            continue
        
        # Clasificar la pieza usando colores de spawn
        tipo = clasificar_pieza_spawn(img, c, x, y, w, h)
        
        if tipo:
            piezas.append((x, y, w, h, tipo))
    
    return piezas

########################################
# DETECTAR PIEZAS EN NEXT/HOLD
########################################

def detectar_piezas_next_hold(img):
    """
    Versión para next/hold que usa NEXT_HOLD_COLORS
    """
    if img.size == 0:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    
    contornos, _ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    piezas = []
    
    for c in contornos:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        tipo = clasificar_pieza_next_hold(img, c, x, y, w, h)
        
        if tipo:
            piezas.append((x, y, w, h, tipo))
    
    return piezas

########################################
# VISUALIZACIÓN
########################################

def visualizar_zonas():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Visualización en vivo (q para salir)")
    print("Zona spawn: ROSADO | Tablero: VERDE | Ocupación: AMARILLO")

    while True:
        screen = capture_screen()

        t = config["tablero"]
        s = config["siguientes"]
        h = config["hold"]

        # Calcular zona spawn (ARRIBA del tablero)
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
        
        # Detectar piezas (usar funciones específicas por zona)
        piezas_spawn = detectar_piezas_spawn(spawn_img)
        piezas_s = detectar_piezas_next_hold(siguientes_img)
        piezas_h = detectar_piezas_next_hold(hold_img)
        
        # Obtener matriz del tablero
        matriz_tablero = obtener_matriz_tablero(tablero_img, t)
        
        # DIBUJADO
        # 1. Zona spawn (rosado)
        cv2.rectangle(
            screen,
            (spawn_region[0], spawn_region[1]),
            (spawn_region[0] + spawn_region[2], spawn_region[1] + spawn_region[3]),
            (255, 105, 180),
            2
        )
        
        # 2. Tablero (verde)
        cv2.rectangle(
            screen,
            (t[0], t[1]),
            (t[0] + t[2], t[1] + t[3]),
            (0, 255, 0),
            2
        )
        
        # 3. Siguientes (azul)
        cv2.rectangle(
            screen,
            (s[0], s[1]),
            (s[0] + s[2], s[1] + s[3]),
            (255, 0, 0),
            2
        )
        
        # 4. Hold (naranja)
        cv2.rectangle(
            screen,
            (h[0], h[1]),
            (h[0] + h[2], h[1] + h[3]),
            (0, 165, 255),
            2
        )
        
        # 5. Dibujar malla en el tablero
        dibujar_malla(screen, t)
        
        # 6. Dibujar ocupación en el tablero (puntos amarillos)
        dibujar_ocupacion(screen, t, matriz_tablero)
        
        # 7. Dibujar piezas en todas las zonas
        dibujar_piezas(screen, spawn_region, piezas_spawn, (255, 105, 180))
        dibujar_piezas(screen, s, piezas_s, (255, 0, 0))
        dibujar_piezas(screen, h, piezas_h, (0, 165, 255))

        cv2.imshow("deteccion tetrio", screen)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

########################################
# VISUALIZACIÓN CON DEPURACIÓN DE SPAWN
########################################

def visualizar_zonas_debug_spawn():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("MODO DEBUG SPAWN - Análisis detallado de la zona spawn")
    print("q para salir")

    while True:
        screen = capture_screen()
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
        spawn_img = capture_screen(spawn_region)
        siguientes_img = capture_screen(s)
        hold_img = capture_screen(h)
        
        # Mostrar análisis de spawn en ventanas separadas
        cv2.imshow("spawn_original", spawn_img)
        
        # Mostrar canal de saturación
        hsv = cv2.cvtColor(spawn_img, cv2.COLOR_BGR2HSV)
        cv2.imshow("spawn_saturation", hsv[:,:,1])
        
        # Umbral de saturación
        _, th_sat = cv2.threshold(hsv[:,:,1], 30, 255, cv2.THRESH_BINARY)
        cv2.imshow("spawn_threshold", th_sat)
        
        # Detectar piezas
        piezas_spawn = detectar_piezas_spawn(spawn_img)
        piezas_s = detectar_piezas_next_hold(siguientes_img)
        piezas_h = detectar_piezas_next_hold(hold_img)
        
        # Crear imagen de debug para spawn
        debug_img = spawn_img.copy()
        
        for (x, y, w, h, tipo) in piezas_spawn:
            # Dibujar bounding box
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_img, tipo, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Mostrar información en consola
            mask = np.zeros(spawn_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            mean_color = cv2.mean(spawn_img, mask=mask)[:3]
            print(f"Pieza {tipo} - Pos: ({x},{y}) - Color: {mean_color}")
        
        cv2.imshow("spawn_detection", debug_img)
        
        # Dibujar en pantalla principal
        cv2.rectangle(screen, (t[0], t[1]), (t[0]+t[2], t[1]+t[3]), (0,255,0), 2)
        cv2.rectangle(screen, 
                     (spawn_region[0], spawn_region[1]),
                     (spawn_region[0]+spawn_region[2], spawn_region[1]+spawn_region[3]),
                     (255,105,180), 2)
        cv2.rectangle(screen, (s[0], s[1]), (s[0]+s[2], s[1]+s[3]), (255,0,0), 2)
        cv2.rectangle(screen, (h[0], h[1]), (h[0]+h[2], h[1]+h[3]), (0,165,255), 2)
        
        # Dibujar piezas
        for (x, y, w, h, tipo) in piezas_spawn:
            px = spawn_region[0] + x + w//2 - 10
            py = spawn_region[1] + y + h//2 + 5
            cv2.putText(screen, tipo, (px, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,105,180), 2)
        
        for (x, y, w, h, tipo) in piezas_s:
            px = s[0] + x + w//2 - 10
            py = s[1] + y + h//2 + 5
            cv2.putText(screen, tipo, (px, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        for (x, y, w, h, tipo) in piezas_h:
            px = h[0] + x + w//2 - 10
            py = h[1] + y + h//2 + 5
            cv2.putText(screen, tipo, (px, py), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        
        cv2.imshow("deteccion_tetrio", screen)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

########################################
# BOT
########################################

def ejecutar_bot():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado")

    while True:
        siguientes = capture_screen(config["siguientes"])
        piezas = detectar_piezas_next_hold(siguientes)

        if len(piezas) > 0:
            keyboard.press('left')
            keyboard.release('left')
            time.sleep(0.1)

            keyboard.press('x')
            keyboard.release('x')
            time.sleep(0.1)

            keyboard.press('space')
            keyboard.release('space')

########################################
# MENÚ
########################################

def menu():
    while True:
        print("\nBOT TETRIO")
        print("1 - Calibrar zonas")
        print("2 - Ejecutar bot")
        print("3 - Ver detección")
        print("4 - Calibrar colores SPAWN")
        print("5 - Calibrar colores NEXT/HOLD")
        print("6 - Modo debug spawn")
        print("7 - Salir")

        op = input("> ")

        if op == "1":
            modo_calibracion()
        elif op == "2":
            ejecutar_bot()
        elif op == "3":
            visualizar_zonas()
        elif op == "4":
            calibrar_colores_spawn()
        elif op == "5":
            calibrar_colores_next_hold()
        elif op == "6":
            visualizar_zonas_debug_spawn()
        elif op == "7":
            break

########################################

if __name__ == "__main__":
    cargar_colores_calibrados()
    menu()

print("exito")