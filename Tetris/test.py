import os
import cv2
import numpy as np
import mss
import json
from pynput.keyboard import Controller, Key, Listener
import time

keyboard = Controller()

CONFIG_FILE = "tetrio_config.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, CONFIG_FILE)

COLOR_TOLERANCE = 30

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
# FORMAS DE PIEZAS
########################################

PIECES = {

"I":[
[[1,1,1,1]],
[[1],[1],[1],[1]]
],

"O":[
[[1,1],
 [1,1]]
],

"T":[
[[0,1,0],
 [1,1,1]],

[[1,0],
 [1,1],
 [1,0]],

[[1,1,1],
 [0,1,0]],

[[0,1],
 [1,1],
 [0,1]]
]
}

bot_activo = False
alt_presionado = False

########################################
# CONTROL ALT+C
########################################

def on_press(key):

    global bot_activo
    global alt_presionado

    if key == Key.alt_l or key == Key.alt_r:
        alt_presionado = True

    try:
        if key.char == 'c' and alt_presionado:
            print("\nBot detenido (ALT+C)")
            bot_activo = False
    except:
        pass


def on_release(key):

    global alt_presionado

    if key == Key.alt_l or key == Key.alt_r:
        alt_presionado = False

########################################
# CAPTURA
########################################

def capturar_region(x, y, w, h):

    with mss.mss() as sct:

        monitor = {
            "top": y,
            "left": x,
            "width": w,
            "height": h
        }

        img = np.array(sct.grab(monitor))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

########################################
# CONTROLES
########################################

def presionar_rotar():

    keyboard.press('x')
    keyboard.release('x')

def hard_drop():

    keyboard.press(Key.space)
    keyboard.release(Key.space)

########################################
# CONFIG
########################################

def cargar_config():

    if not os.path.exists(CONFIG_PATH):
        print("No existe configuración. Ejecuta calibración.")
        return None

    with open(CONFIG_PATH) as f:
        return json.load(f)

########################################
# TABLERO
########################################

def obtener_tamano_celda(region_tablero):

    width = region_tablero[2]
    height = region_tablero[3]

    return width / BOARD_COLS, height / BOARD_ROWS


def obtener_zona_spawn(tablero_region):

    cell_w, cell_h = obtener_tamano_celda(tablero_region)

    return (
        tablero_region[0],
        int(tablero_region[1] - cell_h * 3),
        tablero_region[2],
        int(cell_h * 3)
    )

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

            if np.mean(celda) > 40:
                matriz[fila][col] = 1

    return matriz

########################################
# CLASIFICAR PIEZA
########################################

def clasificar_pieza(img, contorno, x, y, w, h):

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contorno], -1, 255, -1)

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

    contornos,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    piezas = []

    for c in contornos:

        area = cv2.contourArea(c)

        if area < 80:
            continue

        x,y,w,h = cv2.boundingRect(c)

        tipo = clasificar_pieza(img,c,x,y,w,h)

        if tipo is not None:
            piezas.append((x,y,w,h,tipo))

    return piezas

########################################
# LOGICA TETRIS
########################################

def puede_colocar(board, piece, row, col):

    for r in range(len(piece)):
        for c in range(len(piece[0])):

            if piece[r][c] == 0:
                continue

            br = row + r
            bc = col + c

            if br >= BOARD_ROWS or bc < 0 or bc >= BOARD_COLS:
                return False

            if board[br][bc] == 1:
                return False

    return True


def calcular_drop(board, piece, col):

    row = 0

    while puede_colocar(board, piece, row+1, col):

        row += 1

        if row + len(piece) >= BOARD_ROWS:
            break

    return row


def evaluar_tablero(board):

    alturas = []

    for c in range(BOARD_COLS):

        altura = 0

        for r in range(BOARD_ROWS):

            if board[r][c] == 1:
                altura = BOARD_ROWS - r
                break

        alturas.append(altura)

    return sum(alturas)


def mejor_movimiento(board, pieza):

    mejor_score = 1e9
    mejor_col = 0
    mejor_rot = 0

    for rot,forma in enumerate(PIECES[pieza]):

        ancho = len(forma[0])

        for col in range(BOARD_COLS - ancho + 1):

            row = calcular_drop(board, forma, col)

            sim = board.copy()

            for r in range(len(forma)):
                for c in range(ancho):

                    if forma[r][c] == 1:
                        sim[row+r][col+c] = 1

            score = evaluar_tablero(sim)

            if score < mejor_score:

                mejor_score = score
                mejor_col = col
                mejor_rot = rot

    return mejor_col, mejor_rot

########################################
# MOVIMIENTO
########################################

def mover_a_columna(col_objetivo):

    centro = BOARD_COLS // 2

    diff = col_objetivo - centro

    if diff > 0:
        for _ in range(diff):
            keyboard.press(Key.right)
            keyboard.release(Key.right)
            time.sleep(0.02)

    elif diff < 0:
        for _ in range(abs(diff)):
            keyboard.press(Key.left)
            keyboard.release(Key.left)
            time.sleep(0.02)

########################################
# BOT
########################################

def ejecutar_bot():

    global config
    global bot_activo

    config = cargar_config()

    if config is None:
        return

    print("\nBOT INICIADO")
    print("ALT+C para detener el bot")

    tablero = config["tablero"]
    spawn = obtener_zona_spawn(tablero)

    bot_activo = True

    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()

    while bot_activo:

        spawn_img = capturar_region(
            spawn[0],
            spawn[1],
            spawn[2],
            spawn[3]
        )

        piezas = detectar_piezas(spawn_img)

        if len(piezas) == 0:
            continue

        pieza = piezas[0][4]

        if pieza not in PIECES:
            continue

        tablero_img = capturar_region(
            tablero[0],
            tablero[1],
            tablero[2],
            tablero[3]
        )

        grid = obtener_matriz_tablero(tablero_img, tablero)

        columna, rotaciones = mejor_movimiento(grid, pieza)

        for _ in range(rotaciones):
            presionar_rotar()
            time.sleep(0.05)

        mover_a_columna(columna)

        hard_drop()

        time.sleep(0.05)

    listener.stop()

########################################
# MENU
########################################

def menu():

    while True:

        print("\nBOT TETRIO")
        print("1 - Ejecutar bot")
        print("2 - Calibrar zonas")
        print("3 - Salir")

        op = input("> ")

        if op == "1":
            ejecutar_bot()

        elif op == "2":
            print("Calibración no incluida en esta versión")

        elif op == "3":
            break

########################################

if __name__ == "__main__":

    menu()