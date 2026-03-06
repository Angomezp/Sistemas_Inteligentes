import cv2
import numpy as np
import time
import pyautogui
from PIL import ImageGrab
from collections import deque
import threading
from pynput import keyboard, mouse
import json
import tkinter as tk
import os
import sys

# ==========================
#  CONFIGURACION DEL AREA DE JUEGO
# ==========================
GAME_X1, GAME_Y1 = 394, 460
GAME_X2, GAME_Y2 = 1100, 838
GAME_W = GAME_X2 - GAME_X1
GAME_H = GAME_Y2 - GAME_Y1
MITAD_X = GAME_W // 2

# ==========================
#  RUTAS A LAS PLANTILLAS
# ==========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SPRITES_DIR = os.path.join(BASE_DIR, "sprites")

PATH_OJO_MISIONERO = os.path.join(SPRITES_DIR, "misionero.png")
PATH_OJO_CANIBAL   = os.path.join(SPRITES_DIR, "canibal.png")
PATH_OJO_BALSA     = os.path.join(SPRITES_DIR, "balsa.png")
PATH_PANTALLA_FINAL = os.path.join(SPRITES_DIR, "pantalla_final.png")  

# Archivo de configuración para guardar las coordenadas
CONFIG_FILE = os.path.join(BASE_DIR, "area_config.json")

# ==========================
#  VARIABLES GLOBALES
# ==========================
ALT_C_PRESSED = False
plantillas = None

# ==========================
#  FUNCIONES DE CONFIGURACION
# ==========================
def guardar_configuracion(x1, y1, x2, y2):
    config = {
        'GAME_X1': x1,
        'GAME_Y1': y1,
        'GAME_X2': x2,
        'GAME_Y2': y2
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[+] Configuracion guardada en {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"[-] Error al guardar configuracion: {e}")
        return False

def cargar_configuracion():
    global GAME_X1, GAME_Y1, GAME_X2, GAME_Y2, GAME_W, GAME_H, MITAD_X
    if not os.path.exists(CONFIG_FILE):
        return False
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        
        GAME_X1 = config['GAME_X1']
        GAME_Y1 = config['GAME_Y1']
        GAME_X2 = config['GAME_X2']
        GAME_Y2 = config['GAME_Y2']
        GAME_W = GAME_X2 - GAME_X1
        GAME_H = GAME_Y2 - GAME_Y1
        MITAD_X = GAME_W // 2
        
        print(f"[+] Configuracion cargada: ({GAME_X1},{GAME_Y1}) - ({GAME_X2},{GAME_Y2})")
        return True
    except Exception as e:
        print(f"[-] Error al cargar configuracion: {e}")
        return False

# ==========================
#  CALIBRACION CON 2 CLICS
# ==========================
def calibrar_area_con_clics():
    print("\n" + "="*50)
    print("CALIBRACION DEL AREA DE JUEGO")
    print("="*50)
    print("\nInstrucciones:")
    print("  1. Haz clic en la esquina SUPERIOR IZQUIERDA del area de juego")
    print("  2. Haz clic en la esquina INFERIOR DERECHA del area de juego")
    print("\nEsperando clics...")
    
    coords = []
    
    def on_click(x, y, button, pressed):
        if button == mouse.Button.left and pressed:
            coords.append((x, y))
            print(f"   Clic {len(coords)} registrado en: ({int(x)}, {int(y)})")
            if len(coords) >= 2:
                return False
    
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
    
    if len(coords) < 2:
        print("\n[-] No se registraron los 2 clics.")
        return False
    
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    
    nx1 = min(x1, x2)
    ny1 = min(y1, y2)
    nx2 = max(x1, x2)
    ny2 = max(y1, y2)
    
    print(f"\n[+] Area calibrada: ({nx1},{ny1}) - ({nx2},{ny2})")
    print(f"    Ancho: {nx2-nx1} px, Alto: {ny2-ny1} px")
    
    global GAME_X1, GAME_Y1, GAME_X2, GAME_Y2, GAME_W, GAME_H, MITAD_X
    GAME_X1, GAME_Y1, GAME_X2, GAME_Y2 = nx1, ny1, nx2, ny2
    GAME_W = GAME_X2 - GAME_X1
    GAME_H = GAME_Y2 - GAME_Y1
    MITAD_X = GAME_W // 2
    
    guardar_configuracion(GAME_X1, GAME_Y1, GAME_X2, GAME_Y2)
    return True

# ==========================
#  UMBRALES DE CONFIANZA
# ==========================
UMBRAL = 0.73
UMBRAL_BALSA = 0.8
DISTANCIA_MAX_AGRUPAR = 40
DISTANCIA_BALSA = 65
UMBRAL_PANTALLA_FINAL = 0.7

# ==========================
#  OFFSETS
# ==========================
BALSA_CLICK_OFFSET_X = -5
BALSA_CLICK_OFFSET_Y = -5
BALSA_DETECCION_OFFSET_X = -80
BALSA_DETECCION_OFFSET_Y = -75

MISIONERO_CLICK_OFFSET_X = 5
MISIONERO_CLICK_OFFSET_Y = 10
CANIBAL_CLICK_OFFSET_X = -7
CANIBAL_CLICK_OFFSET_Y = 10

# ==========================
#  CARGAR PLANTILLAS
# ==========================
def cargar_plantilla(ruta):
    if not os.path.exists(ruta):
        return None
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

def cargar_todas():
    plantillas = {
        'misionero': cargar_plantilla(PATH_OJO_MISIONERO),
        'canibal': cargar_plantilla(PATH_OJO_CANIBAL),
        'balsa': cargar_plantilla(PATH_OJO_BALSA),
        'pantalla_final': cargar_plantilla(PATH_PANTALLA_FINAL)
    }
    for k, v in plantillas.items():
        if v is None and k != 'pantalla_final':
            print(f"Error: No se pudo cargar {k}")
            return None
    if plantillas['pantalla_final'] is not None:
        print("[+] Plantilla de pantalla final cargada")
    print("Plantillas OK")
    return plantillas

# ==========================
#  DETECCION POR PLANTILLA
# ==========================
def detectar_plantilla(imagen_grande, plantilla, umbral=UMBRAL):
    if plantilla is None:
        return []
    img_gris = cv2.cvtColor(imagen_grande, cv2.COLOR_BGR2GRAY) if len(imagen_grande.shape) == 3 else imagen_grande
    w, h = plantilla.shape[::-1]
    res = cv2.matchTemplate(img_gris, plantilla, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= umbral)
    puntos = [(pt[0] + w//2, pt[1] + h//2) for pt in zip(*loc[::-1])]
    
    unicos = []
    for p in puntos:
        if not any(abs(p[0]-q[0]) < w//2 and abs(p[1]-q[1]) < h//2 for q in unicos):
            unicos.append(p)
    return unicos

def detectar_todos(imagen):
    detecciones = {}
    for clave, plantilla in plantillas.items():
        if plantilla is not None:
            umbral = UMBRAL_BALSA if clave == 'balsa' else UMBRAL
            detecciones[clave] = detectar_plantilla(imagen, plantilla, umbral)
    return detecciones
