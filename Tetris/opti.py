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

COLOR_TOLERANCE = 35

########################################
# CONFIGURACION TABLERO
########################################

BOARD_ROWS = 20
BOARD_COLS = 10

########################################
# COLORES DE PIEZAS
########################################

TETROMINO_COLORS = {
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
    config["spawn"] = calibrar_area("SPAWN")
    config["siguientes"] = calibrar_area("SIGUIENTES")
    config["hold"] = calibrar_area("HOLD")

    with open(CONFIG_PATH,"w") as f:
        json.dump(config,f)

    print("Calibración guardada")

########################################
# CALIBRACIÓN DE COLORES
########################################

def calibrar_colores():

    print("\nCalibración de colores")
    print("Haz click sobre cada pieza cuando se te indique")

    piezas_orden = ["I","J","L","O","S","T","Z"]

    screen = capture_screen()

    colores = {}
    indice = [0]

    def click(event,x,y,flags,param):

        if event == cv2.EVENT_LBUTTONDOWN:

            pieza_actual = piezas_orden[indice[0]]

            color = screen[y,x]

            print(f"Color capturado para {pieza_actual}: {color}")

            colores[pieza_actual] = color.tolist()

            indice[0] += 1

            if indice[0] < len(piezas_orden):
                print(f"Haz click sobre la pieza {piezas_orden[indice[0]]}")
            else:
                print("Todas las piezas capturadas")

    cv2.namedWindow("calibrar_colores")
    cv2.setMouseCallback("calibrar_colores",click)

    print(f"Haz click sobre la pieza {piezas_orden[0]}")

    while True:

        cv2.imshow("calibrar_colores",screen)

        if indice[0] >= len(piezas_orden):
            break

        cv2.waitKey(1)

    cv2.destroyAllWindows()

    if os.path.exists(CONFIG_PATH):

        with open(CONFIG_PATH) as f:
            config = json.load(f)

    else:
        config = {}

    config["colores"] = colores
    config["tolerancia_color"] = COLOR_TOLERANCE

    with open(CONFIG_PATH,"w") as f:
        json.dump(config,f)

    print("Colores calibrados y guardados")

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

    # SOLO filas 1 a 19
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
# DIBUJAR PIEZAS
########################################

def dibujar_piezas(screen,region,piezas,color):

    x_offset = region[0]
    y_offset = region[1]

    for (x,y,w,h,tipo) in piezas:

        px = x + x_offset
        py = y + y_offset

        cv2.putText(
            screen,
            tipo,
            (px,py),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

########################################
# CARGAR COLORES
########################################

def cargar_colores_calibrados():

    global TETROMINO_COLORS
    global COLOR_TOLERANCE

    if not os.path.exists(CONFIG_PATH):
        return

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    if "colores" in config:

        for pieza,color in config["colores"].items():
            TETROMINO_COLORS[pieza] = np.array(color)

    if "tolerancia_color" in config:
        COLOR_TOLERANCE = config["tolerancia_color"]

########################################
# CLASIFICAR PIEZA
########################################

def clasificar_pieza(img, contorno, x, y, w, h):

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask,[contorno],-1,255,-1)

    mean_color = cv2.mean(img, mask=mask)[:3]
    mean_color = np.array(mean_color)

    ratio = w / h if h != 0 else 1

    if ratio > 2.5 or ratio < 0.4:
        return "I"

    if 0.8 < ratio < 1.2:
        dist_o = np.linalg.norm(mean_color - TETROMINO_COLORS["O"])
        if dist_o < COLOR_TOLERANCE:
            return "O"

    mejor = None
    mejor_dist = 1e9

    for pieza, color in TETROMINO_COLORS.items():

        dist = np.linalg.norm(mean_color - color)

        if dist < mejor_dist and dist < COLOR_TOLERANCE:
            mejor_dist = dist
            mejor = pieza

    return mejor

########################################
# DETECTAR PIEZAS
########################################

def detectar_piezas(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(5,5),0)

    _,th = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)

    contornos,_ = cv2.findContours(
        th,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    piezas = []

    for c in contornos:

        area = cv2.contourArea(c)

        if area < 80:
            continue

        x,y,w,h = cv2.boundingRect(c)

        tipo = clasificar_pieza(img,c,x,y,w,h)

        piezas.append((x,y,w,h,tipo))

    return piezas

########################################
# VISUALIZACIÓN
########################################

def visualizar_zonas():

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print("Visualización en vivo (q para salir)")

    while True:

        screen = capture_screen()

        t = config["tablero"]
        sp = config["spawn"]
        s = config["siguientes"]
        h = config["hold"]

        tablero_img = capture_screen(t)
        spawn_img = capture_screen(sp)

        siguientes_img = capture_screen(s)
        hold_img = capture_screen(h)

        matriz_tablero = obtener_matriz_tablero(tablero_img, t)

        piezas_spawn = detectar_piezas(spawn_img)
        piezas_s = detectar_piezas(siguientes_img)
        piezas_h = detectar_piezas(hold_img)

        cv2.rectangle(screen,(t[0],t[1]),(t[0]+t[2],t[1]+t[3]),(0,255,0),2)
        dibujar_malla(screen, t)
        dibujar_ocupacion(screen, t, matriz_tablero)

        cv2.rectangle(screen,(sp[0],sp[1]),(sp[0]+sp[2],sp[1]+sp[3]),(255,105,180),2)
        dibujar_piezas(screen,sp,piezas_spawn,(255,105,180))

        cv2.rectangle(screen,(s[0],s[1]),(s[0]+s[2],s[1]+s[3]),(255,0,0),2)
        dibujar_piezas(screen,s,piezas_s,(255,0,0))

        cv2.rectangle(screen,(h[0],h[1]),(h[0]+h[2],h[1]+h[3]),(0,165,255),2)
        dibujar_piezas(screen,h,piezas_h,(0,165,255))

        cv2.imshow("deteccion tetrio",screen)

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

        spawn = capture_screen(config["spawn"])

        piezas = detectar_piezas(spawn)

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
        print("4 - Calibrar colores")
        print("5 - Salir")

        op = input("> ")

        if op == "1":
            modo_calibracion()

        elif op == "2":
            ejecutar_bot()

        elif op == "3":
            visualizar_zonas()

        elif op == "4":
            calibrar_colores()

        elif op == "5":
            break

########################################

if __name__ == "__main__":

    cargar_colores_calibrados()
    menu()

print("exito")
