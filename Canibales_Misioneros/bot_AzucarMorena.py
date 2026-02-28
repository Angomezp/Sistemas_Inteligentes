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
import signal

# ==========================
#  MANEJADOR DE CTRL+C
# ==========================
def signal_handler(sig, frame):
    print("\n\n[!] Bot cancelado por el usuario (Ctrl+C)")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==========================
#  CONFIGURACION DEL AREA DE JUEGO (valores por defecto)
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

# Archivo de configuración para guardar las coordenadas
CONFIG_FILE = os.path.join(BASE_DIR, "area_config.json")

# ==========================
#  FUNCIONES DE CONFIGURACION
# ==========================
def guardar_configuracion(x1, y1, x2, y2):
    """Guarda la configuración del área en un archivo JSON"""
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
    """Carga la configuración del área desde el archivo JSON"""
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
    """Calibra el área de juego con 2 clics del ratón"""
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

# ==========================
#  OFFSETS PARA BALSA
# ==========================
BALSA_CLICK_OFFSET_X = -5
BALSA_CLICK_OFFSET_Y = -5
BALSA_DETECCION_OFFSET_X = -80
BALSA_DETECCION_OFFSET_Y = -75

# ==========================
#  OFFSETS SEPARADOS PARA CADA TIPO DE PERSONAJE
# ==========================
MISIONERO_CLICK_OFFSET_X = 5
MISIONERO_CLICK_OFFSET_Y = 10
CANIBAL_CLICK_OFFSET_X = -7
CANIBAL_CLICK_OFFSET_Y = 10

# ==========================
#  CARGAR PLANTILLAS
# ==========================
def cargar_plantilla(ruta):
    if not os.path.exists(ruta):
        print(f"Error: {ruta} no encontrado")
        return None
    return cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)

def cargar_todas():
    plantillas = {
        'misionero': cargar_plantilla(PATH_OJO_MISIONERO),
        'canibal': cargar_plantilla(PATH_OJO_CANIBAL),
        'balsa': cargar_plantilla(PATH_OJO_BALSA)
    }
    for k, v in plantillas.items():
        if v is None:
            print(f"Error: No se pudo cargar {k}")
            return None
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

# ==========================
#  AGRUPAR PUNTOS CERCANOS
# ==========================
def agrupar_puntos_cercanos(puntos, distancia_max=DISTANCIA_MAX_AGRUPAR):
    if not puntos:
        return []
    grupos, usados = [], [False] * len(puntos)
    for i, p in enumerate(puntos):
        if usados[i]:
            continue
        grupo, usados[i] = [p], True
        for j, q in enumerate(puntos):
            if not usados[j] and np.linalg.norm(np.array(p)-np.array(q)) < distancia_max:
                grupo.append(q)
                usados[j] = True
        grupos.append((sum(pt[0] for pt in grupo) // len(grupo), 
                       sum(pt[1] for pt in grupo) // len(grupo)))
    return grupos

# ==========================
#  CLASIFICAR PERSONAJES
# ==========================
def clasificar_personajes(misioneros, canibales, centro_deteccion):
    personajes = {
        'izquierda': {'m': [], 'c': []},
        'derecha': {'m': [], 'c': []},
        'balsa': {'m': [], 'c': []}
    }
    
    if centro_deteccion is None:
        for x, y in misioneros:
            if x < MITAD_X:
                personajes['izquierda']['m'].append((x, y))
            else:
                personajes['derecha']['m'].append((x, y))
        for x, y in canibales:
            if x < MITAD_X:
                personajes['izquierda']['c'].append((x, y))
            else:
                personajes['derecha']['c'].append((x, y))
        return personajes
    
    bx, by = centro_deteccion
    for x, y in misioneros:
        if np.linalg.norm([x-bx, y-by]) < DISTANCIA_BALSA:
            personajes['balsa']['m'].append((x, y))
        elif x < MITAD_X:
            personajes['izquierda']['m'].append((x, y))
        else:
            personajes['derecha']['m'].append((x, y))
    
    for x, y in canibales:
        if np.linalg.norm([x-bx, y-by]) < DISTANCIA_BALSA:
            personajes['balsa']['c'].append((x, y))
        elif x < MITAD_X:
            personajes['izquierda']['c'].append((x, y))
        else:
            personajes['derecha']['c'].append((x, y))
    
    return personajes

# ==========================
#  CONVERTIR COORDENADAS
# ==========================
def rel_a_abs(x_rel, y_rel):
    return GAME_X1 + x_rel, GAME_Y1 + y_rel

# ==========================
#  FUNCION DE CLIC MEJORADA
# ==========================
def click(posiciones_rel, count=1, offset=5, desc="", es_balsa=False, es_misionero=False, es_canibal=False):
    for i in range(count):
        if i < len(posiciones_rel):
            x_rel, y_rel = posiciones_rel[i]
            x_abs, y_abs = rel_a_abs(x_rel, y_rel)
            
            if es_balsa:
                x_abs += BALSA_CLICK_OFFSET_X
                y_abs += BALSA_CLICK_OFFSET_Y
            elif es_misionero:
                x_abs += MISIONERO_CLICK_OFFSET_X
                y_abs += MISIONERO_CLICK_OFFSET_Y
            elif es_canibal:
                x_abs += CANIBAL_CLICK_OFFSET_X
                y_abs += CANIBAL_CLICK_OFFSET_Y
            else:
                x_abs += offset
                y_abs += offset
  
            pyautogui.moveTo(x_abs, y_abs, duration=0.1)
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(0.5)
        else:
            break

# ==========================
#  OBTENER ESTADO
# ==========================
def get_game_state(mensaje=""):
    if mensaje:
        print(mensaje)
    
    img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(GAME_X1, GAME_Y1, GAME_X2, GAME_Y2))), cv2.COLOR_RGB2BGR)
    det = detectar_todos(img)
    
    m = agrupar_puntos_cercanos(det.get('misionero', []))[:3]
    c = agrupar_puntos_cercanos(det.get('canibal', []))[:3]
    b = det.get('balsa', [])
    
    balsa_real = b[0] if b else None
    
    if balsa_real:
        click_pos = (balsa_real[0] + BALSA_CLICK_OFFSET_X, balsa_real[1] + BALSA_CLICK_OFFSET_Y)
        centro_det = (click_pos[0] + BALSA_DETECCION_OFFSET_X, click_pos[1] + BALSA_DETECCION_OFFSET_Y)
        boat_side = 0 if balsa_real[0] >= MITAD_X else 1
    else:
        centro_det = None
        boat_side = 0
    
    clasif = clasificar_personajes(m, c, centro_det)
    personajes_balsa = clasif['balsa']['m'] + clasif['balsa']['c']
    
    state = {
        'derecha': (len(clasif['derecha']['m']), len(clasif['derecha']['c'])),
        'izquierda': (len(clasif['izquierda']['m']), len(clasif['izquierda']['c'])),
        'balsa': (len(clasif['balsa']['m']), len(clasif['balsa']['c'])),
        'total_balsa': len(personajes_balsa),
        'boat_side': boat_side,
        'balsa_real': balsa_real,
        'centro_det': centro_det,
        'personajes_balsa': personajes_balsa,
        'todos_misioneros': m,
        'todos_canibales': c,
        'positions_rel': {
            'derecha_m': clasif['derecha']['m'],
            'derecha_c': clasif['derecha']['c'],
            'izquierda_m': clasif['izquierda']['m'],
            'izquierda_c': clasif['izquierda']['c'],
            'balsa_m': clasif['balsa']['m'],
            'balsa_c': clasif['balsa']['c']
        }
    }
    
    return state

# ==========================
#  ENCONTRAR PERSONAJE
# ==========================
def encontrar_personaje(estado, tipo, idx, orilla):
    key = f"{orilla[:3]}_{tipo}"
    personajes = estado['positions_rel'].get(key, [])
    
    if idx < len(personajes):
        return personajes[idx]
    
    if tipo == 'm':
        todos = estado['todos_misioneros']
    else:
        todos = estado['todos_canibales']
    
    if orilla == 'izquierda':
        filtrados = [p for p in todos if p[0] < MITAD_X]
    else:
        filtrados = [p for p in todos if p[0] >= MITAD_X]
    
    if idx < len(filtrados):
        return filtrados[idx]
    
    return None

# ==========================
#  VERIFICAR ESTADO
# ==========================
def verificar_estado(esperado, real, msg):
    if esperado is None:
        return True
    return (esperado[0] == real['derecha'][0] and 
            esperado[1] == real['derecha'][1] and 
            esperado[2] == real['boat_side'])

# ==========================
#  VALIDAR ESTADO
# ==========================
def is_valid_state(m_derecha, c_derecha):
    m_izq, c_izq = 3 - m_derecha, 3 - c_derecha
    if m_derecha > 0 and m_derecha < c_derecha: return False
    if m_izq > 0 and m_izq < c_izq: return False
    return True

# ==========================
#  BFS
# ==========================
def solve_bfs(state):
    start = (state['derecha'][0], state['derecha'][1], state['boat_side'])
    goal = (0, 0, 1)
    
    if start == goal:
        return []
    
    moves = [(1,0), (0,1), (2,0), (0,2), (1,1)]
    move_names = {(1,0): 'M', (0,1): 'C', (2,0): 'MM', (0,2): 'CC', (1,1): 'MC'}
    
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        (m, c, b), path = queue.popleft()
        for dm, dc in moves:
            if b == 0:
                nm, nc, nb = m - dm, c - dc, 1
            else:
                nm, nc, nb = m + dm, c + dc, 0
            
            if 0 <= nm <= 3 and 0 <= nc <= 3 and is_valid_state(nm, nc):
                new_state = (nm, nc, nb)
                if new_state == goal:
                    return path + [move_names[(dm, dc)]]
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [move_names[(dm, dc)]]))
    return None

# ==========================
#  POSICION APROXIMADA
# ==========================
def get_approx_position(bank, idx=0):
    x = GAME_W // 4 if bank == 'izquierda' else 3 * GAME_W // 4
    y = GAME_H // 2 + (idx * 40)
    return (x, y)

# ==========================
#  BAJAR PERSONAJES
# ==========================
def bajar_todos_los_personajes(move, to_bank, estado_actual):
    move_map = {
        'M': [('m', 0)], 'C': [('c', 0)],
        'MM': [('m', 0), ('m', 1)],
        'CC': [('c', 0), ('c', 1)],
        'MC': [('m', 0), ('c', 0)]
    }
    
    persons = move_map.get(move, [])
    temp_state = estado_actual
    
    for intento in range(3):
        balsa_m_list = list(temp_state['positions_rel'].get('balsa_m', []))
        balsa_c_list = list(temp_state['positions_rel'].get('balsa_c', []))
        personajes_balsa = list(temp_state['personajes_balsa'])
        
        if len(personajes_balsa) == 0:
            break
            
        for idx, (typ, person_idx) in enumerate(persons):
            if typ == 'm':
                current_list = balsa_m_list
            else:
                current_list = balsa_c_list
            
            if len(current_list) > 0:
                click_pos = [current_list[0]]
                removed_pos = current_list.pop(0)
                for i, pos in enumerate(personajes_balsa):
                    if pos == removed_pos:
                        personajes_balsa.pop(i)
                        break
            elif len(personajes_balsa) > 0:
                click_pos = [personajes_balsa[0]]
                removed_pos = personajes_balsa.pop(0)
            else:
                break
            
            if typ == 'm':
                click(click_pos, es_misionero=True)
            else:
                click(click_pos, es_canibal=True)
            
            time.sleep(0.5)
            
        temp_state = get_game_state()
        if temp_state['total_balsa'] == 0:
            break
        time.sleep(0.5)
    
    return get_game_state()

# ==========================
#  EJECUTAR MOVIMIENTO
# ==========================
def execute_move(move, estado_actual, estado_esperado_despues):
    if estado_actual['boat_side'] == 0:
        from_bank, to_bank = 'derecha', 'izquierda'
    else:
        from_bank, to_bank = 'izquierda', 'derecha'
    
    move_map = {
        'M': [('m', 0)], 'C': [('c', 0)],
        'MM': [('m', 0), ('m', 1)],
        'CC': [('c', 0), ('c', 1)],
        'MC': [('m', 0), ('c', 0)]
    }
    
    persons = move_map.get(move, [])
    
    for typ, idx in persons:
        pos = encontrar_personaje(estado_actual, typ, idx, from_bank)
        if pos:
            if typ == 'm':
                click([pos], es_misionero=True)
            else:
                click([pos], es_canibal=True)
        else:
            click([get_approx_position(from_bank, idx)])
        time.sleep(0.5)
    
    time.sleep(1)
    
    if estado_actual.get('balsa_real'):
        click([estado_actual['balsa_real']], es_balsa=True)
    else:
        approx = (3 * GAME_W // 4, GAME_H // 2) if estado_actual['boat_side'] == 0 else (GAME_W // 4, GAME_H // 2)
        click([approx], es_balsa=True)
    
    time.sleep(3)
    
    estado_final = bajar_todos_los_personajes(move, to_bank, get_game_state())
    
    if verificar_estado(estado_esperado_despues, estado_final, ""):
        return True, estado_final
    return False, estado_final

# ==========================
#  MOSTRAR AREA
# ==========================
def mostrar_area(duracion=0.8):
    root = tk.Tk()
    root.attributes('-alpha', 0.3, '-topmost', True)
    root.overrideredirect(True)
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
    canvas = tk.Canvas(root, highlightthickness=0, bg='black')
    canvas.pack(fill='both', expand=True)
    canvas.create_rectangle(GAME_X1, GAME_Y1, GAME_X2, GAME_Y2, outline='lime', width=3)
    root.after(int(duracion*1000), root.destroy)
    root.mainloop()

# ==========================
#  VISUALIZADOR
# ==========================
def visualizar_deteccion():
    print("\nVisualizador - ESC para salir")
    while True:
        img = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(GAME_X1, GAME_Y1, GAME_X2, GAME_Y2))), cv2.COLOR_RGB2BGR)
        det = detectar_todos(img)
        
        m = agrupar_puntos_cercanos(det.get('misionero', []))
        c = agrupar_puntos_cercanos(det.get('canibal', []))
        b = det.get('balsa', [])
        
        balsa_real = b[0] if b else None
        
        if balsa_real:
            click_pos = (balsa_real[0] + BALSA_CLICK_OFFSET_X, balsa_real[1] + BALSA_CLICK_OFFSET_Y)
            centro_det = (click_pos[0] + BALSA_DETECCION_OFFSET_X, click_pos[1] + BALSA_DETECCION_OFFSET_Y)
            cv2.circle(img, (int(centro_det[0]), int(centro_det[1])), DISTANCIA_BALSA, (0, 255, 255), 2)
            cv2.circle(img, (int(click_pos[0]), int(click_pos[1])), 5, (0, 255, 0), -1)
            cv2.circle(img, (int(balsa_real[0]), int(balsa_real[1])), 5, (255, 255, 255), -1)
        
        for x, y in m:
            cv2.circle(img, (x, y), 8, (0, 255, 0), 2)
            cv2.putText(img, "M", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for x, y in c:
            cv2.circle(img, (x, y), 8, (0, 0, 255), 2)
            cv2.putText(img, "C", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.line(img, (MITAD_X, 0), (MITAD_X, GAME_H), (255, 255, 255), 2)
        cv2.imshow("Deteccion", img)
        if cv2.waitKey(100) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

# ==========================
#  EJECUTAR BOT
# ==========================
def ejecutar_bot():
    print("\n" + "="*50)
    print("BOT INICIADO - Presiona Ctrl+C para cancelar")
    print("="*50)
    mostrar_area(0.8)
    
    estado = get_game_state("Inicio:")
    if not estado:
        return
    
    if estado['derecha'] == (0,0) and estado['boat_side'] == 1:
        print("Ya resuelto")
        return
    
    plan = solve_bfs(estado)
    if not plan:
        print("Sin solucion")
        return
    
    print(f"Plan: {' -> '.join(plan)}")
    
    m, c, b = estado['derecha'][0], estado['derecha'][1], estado['boat_side']
    
    for i, move in enumerate(plan):
        print(f"\n--- Paso {i+1}/{len(plan)}: {move} ---")
        print(f"Estado actual: ({m},{c},{b})")
        
        delta = {'M':(1,0), 'C':(0,1), 'MM':(2,0), 'CC':(0,2), 'MC':(1,1)}[move]
        if b == 0:
            nm, nc, nb = m - delta[0], c - delta[1], 1
        else:
            nm, nc, nb = m + delta[0], c + delta[1], 0
        
        print(f"Objetivo: ({nm},{nc},{nb})")
        
        ok, nuevo = execute_move(move, estado, (nm, nc, nb))
        if not ok:
            print(f"Error en {move}")
            break
        
        estado = nuevo
        m, c, b = nm, nc, nb
        print(f"Completado: ({m},{c},{b})")
        time.sleep(0.5)
    
    final = get_game_state("Final:")
    if final['derecha'] == (0,0) and final['boat_side'] == 1:
        print("\nMISION CUMPLIDA")
    else:
        print(f"\nFallo: ({final['derecha'][0]},{final['derecha'][1]},{final['boat_side']})")

# ==========================
#  HOTKEY
# ==========================
def on_press(key):
    if key == keyboard.Key.alt_l:
        on_press.alt = True
    if getattr(on_press, 'alt', False) and hasattr(key, 'char') and key.char == 'x':
        threading.Thread(target=ejecutar_bot).start()

def on_release(key):
    if key == keyboard.Key.alt_l:
        on_press.alt = False

on_press.alt = False

# ==========================
#  PROBAR DETECCION
# ==========================
def probar_deteccion():
    estado = get_game_state()
    print(f"\nDer: {estado['derecha'][0]}M {estado['derecha'][1]}C")
    print(f"Izq: {estado['izquierda'][0]}M {estado['izquierda'][1]}C")
    print(f"Balsa: {estado['balsa'][0]}M {estado['balsa'][1]}C")
    if input("\nVer visualizacion? (s/n): ").lower() == 's':
        visualizar_deteccion()

# ==========================
#  MENU PRINCIPAL
# ==========================
def main():
    global plantillas
    print("="*60)
    print("BOT MISIONEROS Y CANIBALES")
    print("="*60)
    print("\nPresiona Ctrl+C en cualquier momento para cancelar la ejecucion")
    
    # Intentar cargar configuracion del JSON (si existe)
    if os.path.exists(CONFIG_FILE):
        if not cargar_configuracion():
            print("\n[-] Error en el archivo de configuracion. Se requiere recalibrar.")
            if not calibrar_area_con_clics():
                print("\n[-] Calibracion fallida. Saliendo...")
                return
    else:
        print("\n[!] No hay archivo de configuracion. Es necesario calibrar.")
        if not calibrar_area_con_clics():
            print("\n[-] Calibracion fallida. Saliendo...")
            return
    
    # Cargar plantillas
    plantillas = cargar_todas()
    if plantillas is None:
        input("Enter para salir...")
        return
    
    while True:
        print("\n" + "-"*40)
        print("MENU PRINCIPAL")
        print("-"*40)
        print("1. Ejecutar bot (ALT+X)")
        print("2. Probar deteccion")
        print("3. Visualizar deteccion")
        print("4. Recalibrar area de juego")
        print("5. Salir")
        
        op = input("\nOpcion: ").strip()
        
        if op == '1':
            print("\nBot listo. Presiona ALT+X para ejecutar")
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            listener.join()
            break
        elif op == '2':
            probar_deteccion()
        elif op == '3':
            visualizar_deteccion()
        elif op == '4':
            if calibrar_area_con_clics():
                print("\n[+] Calibracion completada.")
            else:
                print("\n[-] Calibracion cancelada.")
            input("\nPresiona Enter para continuar...")
        elif op == '5':
            print("\nHasta luego!")
            break
        else:
            print("Opcion no valida")

if __name__ == "__main__":
    main()