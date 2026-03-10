import os
import cv2
import numpy as np
import mss
import json
from pynput.keyboard import Controller, Key
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
    
    spawn_region = obtener_region_spawn(t)
    
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
# OBTENER ZONA SPAWN (3 FILAS ARRIBA + 1 FILAS DENTRO)
########################################

def obtener_region_spawn(region_tablero):
    cell_w, cell_h = obtener_tamano_celda(region_tablero)
    spawn_height = int(cell_h * SPAWN_ROWS)
    spawn_region = (
        region_tablero[0],
        int(region_tablero[1] - cell_h * (SPAWN_ROWS - 1)),  # Empezar 3 filas arriba del tablero),
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
        
        spawn_region = obtener_region_spawn(t)
        
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
        

        spawn_region = obtener_region_spawn(t)
        
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
# FUNCIONES AUXILIARES PARA EL BOT MEJORADO
########################################

# Diccionario único de formas (rotación 0 = horizontal, sentido horario)
FORMAS_PIEZAS = {
    'I': [
        [[1,1,1,1]],               # 0° horizontal
        [[1],[1],[1],[1]]           # 90° vertical
    ],
    'O': [
        [[1,1],[1,1]]               # única
    ],
    'T': [
        [[0,1,0],[1,1,1]],          # 0° tallo arriba
        [[1,0],[1,1],[1,0]],        # 90° tallo derecha
        [[1,1,1],[0,1,0]],          # 180° tallo abajo
        [[0,1],[1,1],[0,1]]         # 270° tallo izquierda
    ],
    'S': [
        [[0,1,1],[1,1,0]],          # 0° horizontal
        [[1,0],[1,1],[0,1]]         # 90° vertical
    ],
    'Z': [
        [[1,1,0],[0,1,1]],          # 0° horizontal
        [[0,1],[1,1],[1,0]]         # 90° vertical
    ],
    'J': [
        [[1,0,0],[1,1,1]],          # 0° tallo arriba izquierda
        [[1,1],[1,0],[1,0]],        # 90° tallo arriba derecha
        [[1,1,1],[0,0,1]],          # 180° tallo abajo derecha
        [[0,1],[0,1],[1,1]]         # 270° tallo abajo izquierda
    ],
    'L': [
        [[0,0,1],[1,1,1]],          # 0° tallo arriba derecha
        [[1,0],[1,0],[1,1]],        # 90° tallo abajo derecha
        [[1,1,1],[1,0,0]],          # 180° tallo abajo izquierda
        [[1,1],[0,1],[0,1]]         # 270° tallo arriba izquierda
    ]
}

def alturas_columna(matriz):
    """Calcula la altura de cada columna (fila desde arriba donde hay un bloque)"""
    alturas = []
    for col in range(BOARD_COLS):
        for fila in range(BOARD_ROWS):
            if matriz[fila][col] == 1:
                alturas.append(BOARD_ROWS - fila)
                break
        else:
            alturas.append(0)
    return alturas

def contar_huecos(matriz):
    """Cuenta los huecos (celdas vacías con al menos un bloque encima)"""
    huecos = 0
    for col in range(BOARD_COLS):
        bloque_encima = False
        for fila in range(BOARD_ROWS):
            if matriz[fila][col] == 1:
                bloque_encima = True
            elif bloque_encima and matriz[fila][col] == 0:
                huecos += 1
    return huecos

def calcular_bumpiness(alturas):
    """Mide la irregularidad del tablero (suma de diferencias entre columnas adyacentes)"""
    bump = 0
    for i in range(len(alturas)-1):
        bump += abs(alturas[i] - alturas[i+1])
    return bump

def eliminar_lineas(matriz):
    """Elimina filas completas y devuelve nueva matriz y número de líneas eliminadas"""
    nuevas_filas = []
    lineas = 0
    for fila in range(BOARD_ROWS):
        if all(matriz[fila][col] == 1 for col in range(BOARD_COLS)):
            lineas += 1
        else:
            nuevas_filas.append(matriz[fila].copy())
    # Añadir filas vacías arriba
    nuevas_filas = [[0]*BOARD_COLS for _ in range(lineas)] + nuevas_filas
    return np.array(nuevas_filas, dtype=int), lineas

def simular_placement(matriz, pieza, rot, col):
    """
    Simula colocar una pieza en la matriz y devuelve el nuevo tablero y líneas eliminadas.
    pieza: tipo de pieza (str)
    rot: índice de rotación (0..n)
    col: columna donde colocar (esquina superior izquierda de la pieza)
    """
    forma = FORMAS_PIEZAS[pieza][rot]
    altura_p = len(forma)
    ancho_p = len(forma[0])

    nueva = matriz.copy()

    # Encontrar fila de caída
    fila_caida = None
    for fila in range(BOARD_ROWS - altura_p + 1):
        colision = False
        for i in range(altura_p):
            for j in range(ancho_p):
                if forma[i][j] == 1:
                    if nueva[fila + i][col + j] == 1:
                        colision = True
                        break
            if colision:
                break
        if colision:
            fila_caida = fila - 1
            break
    if fila_caida is None:
        fila_caida = BOARD_ROWS - altura_p

    if fila_caida < 0:
        return None, 0

    # Colocar la pieza
    for i in range(altura_p):
        for j in range(ancho_p):
            if forma[i][j] == 1:
                nueva[fila_caida + i][col + j] = 1

    nueva, lineas = eliminar_lineas(nueva)
    return nueva, lineas

def puntuar_tablero(matriz):
    """
    Evalúa la bondad del tablero. Mayor puntuación es mejor.
    Factores: menos huecos, menos altura, menos irregularidad.
    """
    alturas = alturas_columna(matriz)
    huecos = contar_huecos(matriz)
    bump = calcular_bumpiness(alturas)
    altura_max = max(alturas) if alturas else 0

    PESO_HUECOS = -12
    PESO_ALTURA = -5
    PESO_BUMP = -0.5

    puntuacion = (PESO_HUECOS * huecos +
                  PESO_ALTURA * altura_max +
                  PESO_BUMP * bump)
    return puntuacion

def mejor_placement(matriz, pieza):
    """
    Encuentra la mejor posición (rotación y columna) para una pieza.
    Devuelve (mejor_puntuacion, mejor_col, mejor_rot)
    Si no hay placement posible, devuelve (-inf, None, None)
    """
    if pieza not in FORMAS_PIEZAS:
        return float('-inf'), None, None

    rotaciones = FORMAS_PIEZAS[pieza]
    mejor_punt = float('-inf')
    mejor_col = None
    mejor_rot = None

    for rot_idx, forma in enumerate(rotaciones):
        ancho_p = len(forma[0])
        for col in range(BOARD_COLS - ancho_p + 1):
            nuevo_tablero, lineas = simular_placement(matriz, pieza, rot_idx, col)
            if nuevo_tablero is None:
                continue
            punt = puntuar_tablero(nuevo_tablero)
            if punt > mejor_punt:
                mejor_punt = punt
                mejor_col = col
                mejor_rot = rot_idx

    return mejor_punt, mejor_col, mejor_rot

def colocar_pieza_mejorada(pieza, columna_spawn_inicial, columna_objetivo, rotacion_objetivo, keyboard, spawn_region, cell_w):
    """
    Coloca la pieza desde su posición actual de spawn hasta la posición objetivo.
    Primero rota (en sentido horario con la flecha arriba), luego detecta la nueva posición y mueve horizontalmente.
    Se asume que la pieza aparece en orientación 0°.
    """
    num_rot = len(FORMAS_PIEZAS[pieza])
    rot_inicial = 0
    rot_necesarias = (rotacion_objetivo - rot_inicial) % num_rot

    print(f"Rotando {rot_necesarias} veces desde rotación inicial")

    # Aplicar rotaciones una por una
    for i in range(rot_necesarias):
        keyboard.press(Key.up)
        time.sleep(0.05)
        keyboard.release(Key.up)
        time.sleep(0.15)  # espera generosa para que la pieza se estabilice

    # Re-detectar la pieza después de rotar (con reintento)
    columna_actual = columna_spawn_inicial
    for intento in range(2):
        time.sleep(0.1)
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
            time.sleep(0.05)
            keyboard.release(Key.right)
            time.sleep(0.05)
    elif desplazamiento < 0:
        for _ in range(abs(desplazamiento)):
            keyboard.press(Key.left)
            time.sleep(0.05)
            keyboard.release(Key.left)
            time.sleep(0.05)

    # Soltar la pieza
    print("Soltando pieza...")
    time.sleep(0.1)
    keyboard.press(Key.space)
    time.sleep(0.05)
    keyboard.release(Key.space)

    return True

########################################
# BOT MEJORADO
########################################

def ejecutar_bot():
    from pynput.keyboard import Key, Controller
    keyboard = Controller()

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Bot iniciado (modo Tetris inteligente)")
    print("Presiona Ctrl+C para detener")

    # Variables de estado
    pieza_actual = None
    pieza_en_hold = None
    pieza_guardada_hold = None  # Para tracking interno
    ultimo_hold_usado = False
    contador_fallos = 0

    try:
        while True:
            # Capturar todas las zonas
            t = config["tablero"]
            s = config["siguientes"]
            h = config["hold"]

            # Calcular zona spawn
            cell_w, cell_h = obtener_tamano_celda(t)

            spawn_region = obtener_region_spawn(t)

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

            # Actualizar estado del hold
            if piezas_hold:
                pieza_en_hold = piezas_hold[0][4]
            else:
                pieza_en_hold = None

            # Ordenar piezas siguientes por posición Y (más próximas primero)
            if piezas_siguientes:
                piezas_siguientes.sort(key=lambda p: p[1])

            # Mostrar información de debug
            if piezas_spawn:
                print(f"\nPieza en spawn: {piezas_spawn[0][4]}")
            if piezas_siguientes:
                print(f"Siguientes (en orden): {[p[4] for p in piezas_siguientes[:4]]}")
            if pieza_en_hold:
                print(f"Hold: {pieza_en_hold}")

            # Si hay pieza en spawn y no hay pieza actual, es nueva pieza
            if piezas_spawn and pieza_actual is None:
                pieza_actual = piezas_spawn[0][4]
                print(f"\n>>> NUEVA PIEZA DETECTADA: {pieza_actual}")

                # Calcular columna actual de la pieza en spawn
                x_spawn = piezas_spawn[0][0]
                columna_spawn = int(round(x_spawn / cell_w))
                columna_spawn = max(0, min(columna_spawn, BOARD_COLS-1))
                print(f"Posición detectada: columna {columna_spawn} (x={x_spawn}px, cell_w={cell_w:.1f})")

                # Obtener próximas piezas
                proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []

                # Evaluar opciones
                punt_actual, col_objetivo, rot_objetivo = mejor_placement(matriz_tablero, pieza_actual)
                puede_colocar_actual = col_objetivo is not None

                # Opción de hold
                opcion_hold = None
                punt_hold = float('-inf')
                if not ultimo_hold_usado:
                    if pieza_en_hold is None:
                        if proximas_piezas:
                            punt_sig, col_sig, rot_sig = mejor_placement(matriz_tablero, proximas_piezas[0])
                            if col_sig is not None:
                                opcion_hold = 'guardar'
                                punt_hold = punt_sig
                    else:
                        punt_swap, col_swap, rot_swap = mejor_placement(matriz_tablero, pieza_en_hold)
                        if col_swap is not None:
                            opcion_hold = 'swap'
                            punt_hold = punt_swap

                # Decidir acción
                if puede_colocar_actual and punt_actual >= punt_hold:
                    print(f"Colocando {pieza_actual} (puntuación: {punt_actual:.1f})")
                    exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                    keyboard, spawn_region, cell_w)
                    if exito:
                        print(f"Pieza {pieza_actual} colocada")
                        ultimo_hold_usado = False
                    else:
                        print(f"Error colocando {pieza_actual}")
                        contador_fallos += 1
                    pieza_actual = None

                elif punt_hold > float('-inf') and punt_hold > punt_actual:
                    if opcion_hold == 'guardar':
                        print(f"Guardando {pieza_actual} en hold (siguiente {proximas_piezas[0]} puntúa {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        pieza_actual = None
                        time.sleep(0.3)
                    elif opcion_hold == 'swap':
                        print(f"Intercambiando: {pieza_actual} con {pieza_en_hold} (puntuación {punt_hold:.1f})")
                        keyboard.press(Key.shift)
                        time.sleep(0.05)
                        keyboard.release(Key.shift)
                        ultimo_hold_usado = True
                        time.sleep(0.3)
                        pieza_actual = None
                    else:
                        print("Error en opción de hold")
                        pieza_actual = None
                else:
                    print(f"No se puede colocar {pieza_actual} y hold no disponible o no mejora")
                    if puede_colocar_actual:
                        exito = colocar_pieza_mejorada(pieza_actual, columna_spawn, col_objetivo, rot_objetivo,
                                                        keyboard, spawn_region, cell_w)
                        if exito:
                            print(f"Pieza {pieza_actual} colocada (forzada)")
                        else:
                            contador_fallos += 1
                    else:
                        print(f"No hay lugar para {pieza_actual}, esperando...")
                    pieza_actual = None

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nBot detenido")
    except Exception as e:
        print(f"Error en el bot: {e}")
        import traceback
        traceback.print_exc()


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