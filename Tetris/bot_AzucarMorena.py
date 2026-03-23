import os
import cv2
import numpy as np
import mss
import json
from pynput.keyboard import Controller, Key
import time


class bot_AzucarMorena:
    def __init__(self):
        # teclado
        self.keyboard = Controller()

        # Paths y configuración
        self.CONFIG_FILE = "tetrio_config.json"
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_PATH = os.path.join(self.BASE_DIR, self.CONFIG_FILE)
        self.COORDENADAS_PUNTAJE = (308, 447, 363, 420)
        self.SCREENSHOT_PATH = os.path.join(self.BASE_DIR, "azucar_morena_bot.png")

        # Tolerancia de color
        self.COLOR_TOLERANCE = 30

        # Configuracion tablero
        self.BOARD_ROWS = 20
        self.BOARD_COLS = 10
        self.SPAWN_ROWS = 7
        self.INSIDE_SPAWN_ROWS = 4

        # Colores por zona (se cargan del json con cargar_colores_calibrados)
        self.SPAWN_COLORS = {}
        self.NEXT_HOLD_COLORS = {}

        # Formas de piezas
        self.FORMAS_PIEZAS = {
            'I': [
                [[1,1,1,1]],
                [[1],[1],[1],[1]]
            ],
            'O': [
                [[1,1],[1,1]]
            ],
            'T': [
                [[0,1,0],[1,1,1]],
                [[1,0],[1,1],[1,0]],
                [[1,1,1],[0,1,0]],
                [[0,1],[1,1],[0,1]]
            ],
            'S': [
                [[0,1,1],[1,1,0]],
                [[1,0],[1,1],[0,1]]
            ],
            'Z': [
                [[1,1,0],[0,1,1]],
                [[0,1],[1,1],[1,0]]
            ],
            'J': [
                [[1,0,0],[1,1,1]],
                [[1,1],[1,0],[1,0]],
                [[1,1,1],[0,0,1]],
                [[0,1],[0,1],[1,1]]
            ],
            'L': [
                [[0,0,1],[1,1,1]],
                [[1,0],[1,0],[1,1]],
                [[1,1,1],[1,0,0]],
                [[1,1],[0,1],[0,1]]
            ]
        }

    # ----------------------- utilidades de captura y calibración -----------------------
    def capturar_pantalla(self, region=None):
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

    def dibujar_rectangulo(self, event, x, y, flags, param):
        if not hasattr(self, '_drawing'):
            self._drawing = False
            self._ix = -1
            self._iy = -1
            self._rect = None

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._ix, self._iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                img_copy = param.copy()
                cv2.rectangle(img_copy,(self._ix,self._iy),(x,y),(0,255,0),2)
                cv2.imshow("calibracion",img_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            self._rect = (self._ix,self._iy,x-self._ix,y-self._iy)
            cv2.rectangle(param,(self._ix,self._iy),(x,y),(0,255,0),2)
            cv2.imshow("calibracion",param)

    def calibrar_area(self, nombre):
        time.sleep(0.5)
        screen = self.capturar_pantalla()
        self._rect = None

        cv2.namedWindow("calibracion")
        cv2.setMouseCallback("calibracion", self.dibujar_rectangulo, screen)

        print(f"\nSelecciona el area de {nombre}")
        print("Arrastra con el mouse y presiona 's'")

        while True:
            cv2.imshow("calibracion",screen)
            key = cv2.waitKey(1)
            if key == ord('s') and self._rect is not None:
                break

        cv2.destroyAllWindows()
        return self._rect

    def modo_calibracion(self):
        config = {}
        config["tablero"] = self.calibrar_area("TABLERO")
        config["siguientes"] = self.calibrar_area("SIGUIENTES")
        config["hold"] = self.calibrar_area("HOLD")

        with open(self.CONFIG_PATH,"w") as f:
            json.dump(config,f)

        print("Calibración guardada")

    # ----------------------- calibración colores -----------------------
    def calibrar_colores_spawn(self):
        print("\nCalibración de colores - ZONA SPAWN")
        print("Haz click sobre cada pieza cuando se te indique (asegúrate de que esté en la zona de spawn)")

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        t = config["tablero"]
        cell_w, cell_h = self.obtener_tamano_celda(t)
        spawn_region = self.obtener_region_spawn(t)
        screen = self.capturar_pantalla(spawn_region)

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

        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH) as f:
                config = json.load(f)
        else:
            config = {}

        if "colores" not in config:
            config["colores"] = {}

        config["colores"]["spawn"] = colores
        config["tolerancia_color"] = self.COLOR_TOLERANCE

        with open(self.CONFIG_PATH,"w") as f:
            json.dump(config,f, indent=4)

        print("Colores de spawn calibrados y guardados")

    def calibrar_colores_next_hold(self):
        print("\nCalibración de colores - ZONA NEXT/HOLD")
        print("Haz click sobre cada pieza cuando se te indique (puedes usar la zona de next o hold)")

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        screen = self.capturar_pantalla(config["siguientes"])

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

        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH) as f:
                config = json.load(f)
        else:
            config = {}

        if "colores" not in config:
            config["colores"] = {}

        config["colores"]["next_hold"] = colores
        config["tolerancia_color"] = self.COLOR_TOLERANCE

        with open(self.CONFIG_PATH,"w") as f:
            json.dump(config,f, indent=4)

        print("Colores de next/hold calibrados y guardados")

    # ----------------------- tablero y detección -----------------------
    def obtener_tamano_celda(self, region_tablero):
        width = region_tablero[2]
        height = region_tablero[3]
        cell_w = width / self.BOARD_COLS
        cell_h = height / self.BOARD_ROWS
        return cell_w, cell_h

    def obtener_region_spawn(self, region_tablero):
        cell_w, cell_h = self.obtener_tamano_celda(region_tablero)
        spawn_height = int(cell_h * self.SPAWN_ROWS)
        spawn_region = (
            region_tablero[0],
            int(region_tablero[1] - cell_h * (self.SPAWN_ROWS - self.INSIDE_SPAWN_ROWS)),
            region_tablero[2],
            spawn_height
        )
        return spawn_region

    def obtener_matriz_tablero(self, tablero_img, region_tablero):
        matriz = np.zeros((self.BOARD_ROWS, self.BOARD_COLS), dtype=int)
        gray = cv2.cvtColor(tablero_img, cv2.COLOR_BGR2GRAY)
        cell_w, cell_h = self.obtener_tamano_celda(region_tablero)

        for fila in range(self.BOARD_ROWS):
            for col in range(self.BOARD_COLS):
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

    def dibujar_malla(self, screen, region):
        cell_w, cell_h = self.obtener_tamano_celda(region)
        x_offset = region[0]
        y_offset = region[1]

        for c in range(self.BOARD_COLS + 1):
            x = int(x_offset + c * cell_w)
            cv2.line(
                screen,
                (x, y_offset),
                (x, int(y_offset + self.BOARD_ROWS * cell_h)),
                (255,255,255),
                1
            )

        for r in range(1, self.BOARD_ROWS):
            y = int(y_offset + r * cell_h)
            cv2.line(
                screen,
                (x_offset, y),
                (int(x_offset + self.BOARD_COLS * cell_w), y),
                (255,255,255),
                1
            )

    def dibujar_ocupacion(self, screen, region, matriz):
        cell_w, cell_h = self.obtener_tamano_celda(region)
        x_offset = region[0]
        y_offset = region[1]

        for r in range(self.BOARD_ROWS):
            for c in range(self.BOARD_COLS):
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

    def dibujar_piezas(self, screen, region, piezas, color, offset_adicional=(0,0)):
        x_offset = region[0] + offset_adicional[0]
        y_offset = region[1] + offset_adicional[1]
    
        for (x, y, w, h, tipo) in piezas:
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

    def cargar_colores_calibrados(self):
        if not os.path.exists(self.CONFIG_PATH):
            return

        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        if "colores" in config:
            if "spawn" in config["colores"]:
                for pieza, color in config["colores"]["spawn"].items():
                    self.SPAWN_COLORS[pieza] = np.array(color)
                print("Colores de spawn cargados")
        
            if "next_hold" in config["colores"]:
                for pieza, color in config["colores"]["next_hold"].items():
                    self.NEXT_HOLD_COLORS[pieza] = np.array(color)
                print("Colores de next/hold cargados")

        if "tolerancia_color" in config:
            self.COLOR_TOLERANCE = config["tolerancia_color"]

    def clasificar_pieza_spawn(self, img, contorno, x, y, w, h):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        mean_color = cv2.mean(img, mask=mask)[:3]
        mean_color = np.array(mean_color)
        ratio = w / h if h != 0 else 1

        if ratio > 2.0:
            dist_i = np.linalg.norm(mean_color - self.SPAWN_COLORS["I"])
            if dist_i < self.COLOR_TOLERANCE * 1.5:
                return "I"

        mejor = None
        mejor_dist = self.COLOR_TOLERANCE
        for pieza, color in self.SPAWN_COLORS.items():
            dist = np.linalg.norm(mean_color - color)
            if pieza in ["T", "L", "J", "Z"]:
                if 0.7 < ratio < 1.8:
                    dist *= 0.9
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = pieza
        return mejor

    def clasificar_pieza_next_hold(self, img, contorno, x, y, w, h):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contorno], -1, 255, -1)
        mean_color = cv2.mean(img, mask=mask)[:3]
        mean_color = np.array(mean_color)
        ratio = w / h if h != 0 else 1

        if ratio > 2.0:
            dist_i = np.linalg.norm(mean_color - self.NEXT_HOLD_COLORS["I"])
            if dist_i < self.COLOR_TOLERANCE * 1.5:
                return "I"

        mejor = None
        mejor_dist = self.COLOR_TOLERANCE
        for pieza, color in self.NEXT_HOLD_COLORS.items():
            dist = np.linalg.norm(mean_color - color)
            if pieza in ["T", "L", "J", "Z"]:
                if 0.7 < ratio < 1.8:
                    dist *= 0.9
            if dist < mejor_dist:
                mejor_dist = dist
                mejor = pieza
        return mejor

    def detectar_piezas_spawn(self, img):
        if img.size == 0:
            return []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        _, th_sat = cv2.threshold(saturation, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        th_sat = cv2.morphologyEx(th_sat, cv2.MORPH_OPEN, kernel)
        th_sat = cv2.morphologyEx(th_sat, cv2.MORPH_CLOSE, kernel)
        contornos, _ = cv2.findContours(th_sat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        piezas = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 or h < 5:
                continue
            tipo = self.clasificar_pieza_spawn(img, c, x, y, w, h)
            if tipo:
                piezas.append((x, y, w, h, tipo))
        return piezas

    def detectar_piezas_next_hold(self, img):
        if img.size == 0:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, th = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
        contornos, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        piezas = []
        for c in contornos:
            area = cv2.contourArea(c)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(c)
            tipo = self.clasificar_pieza_next_hold(img, c, x, y, w, h)
            if tipo:
                piezas.append((x, y, w, h, tipo))
        return piezas

    def visualizar_zonas(self):
        with open(self.CONFIG_PATH) as f:
            config = json.load(f)

        print("Visualización en vivo (q para salir)")
        print("Zona spawn: ROSADO | Tablero: VERDE | Ocupación: AMARILLO")

        while True:
            screen = self.capturar_pantalla()
            t = config["tablero"]
            s = config["siguientes"]
            h = config["hold"]
            spawn_region = self.obtener_region_spawn(t)
            tablero_img = self.capturar_pantalla(t)
            spawn_img = self.capturar_pantalla(spawn_region)
            siguientes_img = self.capturar_pantalla(s)
            hold_img = self.capturar_pantalla(h)
            piezas_spawn = self.detectar_piezas_spawn(spawn_img)
            piezas_s = self.detectar_piezas_next_hold(siguientes_img)
            piezas_h = self.detectar_piezas_next_hold(hold_img)
            matriz_tablero = self.obtener_matriz_tablero(tablero_img, t)

            cv2.rectangle(screen, (spawn_region[0], spawn_region[1]), (spawn_region[0] + spawn_region[2], spawn_region[1] + spawn_region[3]), (255, 105, 180), 2)
            cv2.rectangle(screen, (t[0], t[1]), (t[0] + t[2], t[1] + t[3]), (0, 255, 0), 2)
            cv2.rectangle(screen, (s[0], s[1]), (s[0] + s[2], s[1] + s[3]), (255, 0, 0), 2)
            cv2.rectangle(screen, (h[0], h[1]), (h[0] + h[2], h[1] + h[3]), (0, 165, 255), 2)
            self.dibujar_malla(screen, t)
            self.dibujar_ocupacion(screen, t, matriz_tablero)
            self.dibujar_piezas(screen, spawn_region, piezas_spawn, (255, 105, 180))
            self.dibujar_piezas(screen, s, piezas_s, (255, 0, 0))
            self.dibujar_piezas(screen, h, piezas_h, (0, 165, 255))
            cv2.imshow("deteccion tetrio", screen)
            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()


    # ----------------------- funciones auxiliares del bot (simulación, heurísticas) -----------------------
    def alturas_columna(self, matriz):
        alturas = []
        for col in range(self.BOARD_COLS):
            for fila in range(self.BOARD_ROWS):
                if matriz[fila][col] == 1:
                    alturas.append(self.BOARD_ROWS - fila)
                    break
            else:
                alturas.append(0)
        return alturas

    def contar_huecos(self, matriz):
        huecos = 0
        for col in range(self.BOARD_COLS):
            bloque_encima = False
            for fila in range(self.BOARD_ROWS):
                if matriz[fila][col] == 1:
                    bloque_encima = True
                elif bloque_encima and matriz[fila][col] == 0:
                    huecos += 1
        return huecos

    def calcular_bumpiness(self, alturas):
        bump = 0
        for i in range(len(alturas)-1):
            bump += abs(alturas[i] - alturas[i+1])
        return bump

    def eliminar_lineas(self, matriz):
        nuevas_filas = []
        lineas = 0
        for fila in range(self.BOARD_ROWS):
            if all(matriz[fila][col] == 1 for col in range(self.BOARD_COLS)):
                lineas += 1
            else:
                nuevas_filas.append(matriz[fila].copy())
        nuevas_filas = [[0]*self.BOARD_COLS for _ in range(lineas)] + nuevas_filas
        return np.array(nuevas_filas, dtype=int), lineas

    def simular_placement(self, matriz, pieza, rot, col):
        forma = self.FORMAS_PIEZAS[pieza][rot]
        altura_p = len(forma)
        ancho_p = len(forma[0])
        nueva = matriz.copy()
        fila_caida = None
        for fila in range(self.BOARD_ROWS - altura_p + 1):
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
            fila_caida = self.BOARD_ROWS - altura_p
        if fila_caida < 0:
            return None, 0
        for i in range(altura_p):
            for j in range(ancho_p):
                if forma[i][j] == 1:
                    nueva[fila_caida + i][col + j] = 1
        nueva, lineas = self.eliminar_lineas(nueva)
        return nueva, lineas

    def calcular_nivel(self, lineas):
        if lineas < 3:
            return 1
        else:
            nivel = 1
            acum = 0
            meta = 3
            while lineas >= acum + meta:
                acum += meta
                nivel += 1
                meta += 2
            return nivel

    def obtener_tiempos_hold(self, nivel, usar_rapida):
        if nivel <= 5:
            factor = 1.0
        elif nivel == 6:
            factor = 0.9
        elif nivel == 7:
            factor = 0.8
        else:
            factor = 0.7
        if usar_rapida:
            factor *= 0.9
        return {
            'hold_press': max(0.04 * factor, 0.015),
            'hold_release_delay': max(0.12 * factor, 0.05)
        }

    def guardar_screenshot_pantalla(self):
        screen = self.capturar_pantalla(self.COORDENADAS_PUNTAJE)
        cv2.imwrite(self.SCREENSHOT_PATH, screen)

    def puntuar_tablero(self, matriz, lineas_eliminadas=0):
        alturas = self.alturas_columna(matriz)
        huecos = self.contar_huecos(matriz)
        bump = self.calcular_bumpiness(alturas)
        altura_max = max(alturas) if alturas else 0
        nivel = self.calcular_nivel(lineas_eliminadas)
        if altura_max == 0:
            return 10000
        PESO_LINEAS = 500
        PESO_HUECOS = -25
        PESO_ALTURA = -10
        PESO_BUMP = -2
        if altura_max >=2 and altura_max <= 5:
            PESO_ALTURA = -5
            PESO_HUECOS = -40
        else:
            PESO_ALTURA *= altura_max
        if lineas_eliminadas == 4:
            bonus = 2000
        elif lineas_eliminadas == 3:
            bonus = 1000
        else:
            bonus = 0
        if nivel == 7:
            PESO_ALTURA *= 1.5
        elif nivel >= 8:
            PESO_ALTURA *= nivel
        puntuacion = (PESO_LINEAS * lineas_eliminadas + PESO_HUECOS * huecos + PESO_ALTURA * altura_max + PESO_BUMP * bump + bonus)
        return puntuacion

    def mejor_placement(self, matriz, pieza):
        if pieza not in self.FORMAS_PIEZAS:
            return float('-inf'), None, None
        rotaciones = self.FORMAS_PIEZAS[pieza]
        mejor_punt = float('-inf')
        mejor_col = None
        mejor_rot = None
        for rot_idx, forma in enumerate(rotaciones):
            ancho_p = len(forma[0])
            for col in range(self.BOARD_COLS - ancho_p + 1):
                nuevo_tablero, lineas = self.simular_placement(matriz, pieza, rot_idx, col)
                if nuevo_tablero is None:
                    continue
                punt = self.puntuar_tablero(nuevo_tablero, lineas)
                if punt > mejor_punt:
                    mejor_punt = punt
                    mejor_col = col
                    mejor_rot = rot_idx
        return mejor_punt, mejor_col, mejor_rot

    def mejor_placement_rapido(self, matriz, pieza, nivel=7):
        if pieza not in self.FORMAS_PIEZAS:
            return float('-inf'), None, None
        rotaciones = self.FORMAS_PIEZAS[pieza]
        alturas_actuales = self.alturas_columna(matriz)
        if nivel >= 8:
            num_rot = 2
            num_cols = 5
        else:
            num_rot = 3
            num_cols = self.BOARD_COLS
        puntajes_rotacion = []
        for rot_idx, forma in enumerate(rotaciones):
            ancho_p = len(forma[0])
            altura_p = len(forma)
            mejor_altura = self.BOARD_ROWS + 1
            for col in range(self.BOARD_COLS - ancho_p + 1):
                fila_caida = self.BOARD_ROWS - altura_p
                for i in range(altura_p):
                    for j in range(ancho_p):
                        if forma[i][j] == 1:
                            for f in range(fila_caida, -1, -1):
                                if matriz[f][col + j] == 1:
                                    fila_caida = min(fila_caida, f - i - 1)
                                    break
                altura_final = self.BOARD_ROWS - fila_caida
                if altura_final < mejor_altura:
                    mejor_altura = altura_final
            puntajes_rotacion.append((rot_idx, mejor_altura))
        puntajes_rotacion.sort(key=lambda x: x[1])
        rotaciones_a_evaluar = [r[0] for r in puntajes_rotacion[:num_rot]]
        mejor_punt = float('-inf')
        mejor_col = None
        mejor_rot = None
        for rot_idx in rotaciones_a_evaluar:
            forma = rotaciones[rot_idx]
            ancho_p = len(forma[0])
            columnas_posibles = list(range(self.BOARD_COLS - ancho_p + 1))
            if nivel == 7:
                columnas_a_evaluar = columnas_posibles
            else:
                puntajes_col = []
                for col in columnas_posibles:
                    alturas_afectadas = [alturas_actuales[col + j] for j in range(ancho_p)]
                    max_altura = max(alturas_afectadas) if alturas_afectadas else 0
                    puntajes_col.append((col, max_altura))
                puntajes_col.sort(key=lambda x: x[1])
                columnas_a_evaluar = [c[0] for c in puntajes_col[:num_cols]]
            for col in columnas_a_evaluar:
                nuevo_tablero, lineas = self.simular_placement(matriz, pieza, rot_idx, col)
                if nuevo_tablero is None:
                    continue
                punt = self.puntuar_tablero(nuevo_tablero, lineas)
                if punt > mejor_punt:
                    mejor_punt = punt
                    mejor_col = col
                    mejor_rot = rot_idx
        if mejor_col is None:
            rotaciones_restantes = [i for i in range(len(rotaciones)) if i not in rotaciones_a_evaluar]
            for rot_idx in rotaciones_restantes:
                forma = rotaciones[rot_idx]
                ancho_p = len(forma[0])
                for col in range(self.BOARD_COLS - ancho_p + 1):
                    nuevo_tablero, lineas = self.simular_placement(matriz, pieza, rot_idx, col)
                    if nuevo_tablero is None:
                        continue
                    punt = self.puntuar_tablero(nuevo_tablero, lineas)
                    if punt > mejor_punt:
                        mejor_punt = punt
                        mejor_col = col
                        mejor_rot = rot_idx
        return mejor_punt, mejor_col, mejor_rot

    def colocar_pieza_mejorada(self, pieza, columna_spawn_inicial, columna_objetivo, rotacion_objetivo, keyboard, spawn_region, cell_w, nivel, usar_estrategia_rapida=False):
        tiempos_base = {
            'pulsacion': 0.025,
            'post_pulsacion': 0.025,
            'post_rotacion':0.03,
            'pre_soltar': 0.04,
            'reintento': 0.15
        }
        if nivel <= 5:
            factor = 1.0
            factor_reintento = 1.0
        elif nivel == 6:
            factor = 0.9
            factor_reintento = 0.9
        elif nivel == 7:
            factor = 0.8
            factor_reintento = 0.8
        else:
            factor = 0.5
            factor_reintento = 0.5
        t_puls = max(tiempos_base['pulsacion'] * factor, 0.002)
        t_post_puls = max(tiempos_base['post_pulsacion'] * factor, 0.002)
        t_post_rot = max(tiempos_base['post_rotacion'] * factor, 0.0025)
        t_pre_soltar = max(tiempos_base['pre_soltar'] * factor, 0.0035)
        t_reintento = max(tiempos_base['reintento'] * factor_reintento, 0.015)

        num_rot = len(self.FORMAS_PIEZAS[pieza])
        if num_rot == 4:
            if rotacion_objetivo == 1:
                tecla = 'x'
                pulsaciones = 1
            elif rotacion_objetivo == 2:
                tecla = 'a'
                pulsaciones = 1
            elif rotacion_objetivo == 3:
                tecla = 'z'
                pulsaciones = 1
            else:
                pulsaciones = 0
        elif num_rot == 2:
            if rotacion_objetivo == 1:
                tecla = 'x'
                pulsaciones = 1
            else:
                pulsaciones = 0
        else:
            pulsaciones = 0
        if pulsaciones > 0:
            keyboard.press(tecla)
            time.sleep(t_puls)
            keyboard.release(tecla)
            time.sleep(t_post_rot)
        columna_actual = columna_spawn_inicial
        for intento in range(2):
            time.sleep(t_reintento)
            spawn_img = self.capturar_pantalla(spawn_region)
            piezas_detectadas = self.detectar_piezas_spawn(spawn_img)
            if piezas_detectadas:
                x_spawn = piezas_detectadas[0][0]
                columna_actual = int(round(x_spawn / cell_w))
                columna_actual = max(0, min(columna_actual, self.BOARD_COLS-1))
                break
        desplazamiento = columna_objetivo - columna_actual
        if desplazamiento > 0:
            for _ in range(desplazamiento):
                keyboard.press(Key.right)
                time.sleep(t_puls)
                keyboard.release(Key.right)
                time.sleep(t_post_puls)
        elif desplazamiento < 0:
            for _ in range(abs(desplazamiento)):
                keyboard.press(Key.left)
                time.sleep(t_puls)
                keyboard.release(Key.left)
                time.sleep(t_post_puls)
        time.sleep(t_pre_soltar)
        keyboard.press(Key.space)
        time.sleep(t_puls)
        keyboard.release(Key.space)
        return True

    # ----------------------- loop principal (compute) -----------------------
    def compute(self):
        keyboard = Controller()
        with open(self.CONFIG_PATH) as f:
            config = json.load(f)
        print("Bot iniciado (modo Tetris con énfasis en limpiar el tablero)")
        print("Niveles 1-6: Estrategia robusta")
        print("Niveles 7+: Estrategia rápida")
        print("Presiona Ctrl+C para detener")
        pieza_actual = None
        pieza_en_hold = None
        ultimo_hold_usado = False
        contador_fallos = 0
        lineas_totales = 0
        nivel = 1
        try:
            while True:
                t = config["tablero"]
                s = config["siguientes"]
                h = config["hold"]
                cell_w, cell_h = self.obtener_tamano_celda(t)
                spawn_region = self.obtener_region_spawn(t)
                tablero_img = self.capturar_pantalla(t)
                spawn_img = self.capturar_pantalla(spawn_region)
                siguientes_img = self.capturar_pantalla(s)
                hold_img = self.capturar_pantalla(h)
                matriz_tablero = self.obtener_matriz_tablero(tablero_img, t)
                piezas_spawn = self.detectar_piezas_spawn(spawn_img)
                piezas_siguientes = self.detectar_piezas_next_hold(siguientes_img)
                piezas_hold = self.detectar_piezas_next_hold(hold_img)
                if piezas_hold:
                    pieza_en_hold = piezas_hold[0][4]
                else:
                    pieza_en_hold = None
                if piezas_siguientes:
                    piezas_siguientes.sort(key=lambda p: p[1])
                    alturas = self.alturas_columna(matriz_tablero)
                    max_altura = max(alturas) if alturas else 0
                    if max_altura == self.BOARD_ROWS:
                        self.guardar_screenshot_pantalla()
                if piezas_spawn and pieza_actual is None:
                    pieza_actual = piezas_spawn[0][4]
                    nivel = self.calcular_nivel(lineas_totales)
                    usar_estrategia_rapida = nivel >= 7
                    x_spawn = piezas_spawn[0][0]
                    columna_spawn = int(round(x_spawn / cell_w))
                    columna_spawn = max(0, min(columna_spawn, self.BOARD_COLS - 1))
                    proximas_piezas = [p[4] for p in piezas_siguientes[:3]] if piezas_siguientes else []
                    if usar_estrategia_rapida:
                        punt_actual, col_obj, rot_obj = self.mejor_placement_rapido(matriz_tablero, pieza_actual, nivel)
                    else:
                        punt_actual, col_obj, rot_obj = self.mejor_placement(matriz_tablero, pieza_actual)
                    puede_colocar_actual = col_obj is not None
                    punt_hold = float('-inf')
                    hold_accion = None
                    nueva_pieza_hold = None
                    col_hold_obj = None
                    rot_hold_obj = None
                    lineas_hold = 0
                    if not ultimo_hold_usado:
                        if pieza_en_hold is None:
                            if proximas_piezas:
                                nueva_pieza = proximas_piezas[0]
                                if usar_estrategia_rapida:
                                    punt_tmp, col_tmp, rot_tmp = self.mejor_placement_rapido(matriz_tablero, nueva_pieza, nivel)
                                else:
                                    punt_tmp, col_tmp, rot_tmp = self.mejor_placement(matriz_tablero, nueva_pieza)
                                if col_tmp is not None:
                                    nuevo_tablero, lineas = self.simular_placement(matriz_tablero, nueva_pieza, rot_tmp, col_tmp)
                                    punt_hold = self.puntuar_tablero(nuevo_tablero, lineas)
                                    hold_accion = 'guardar'
                                    nueva_pieza_hold = nueva_pieza
                                    col_hold_obj = col_tmp
                                    rot_hold_obj = rot_tmp
                                    lineas_hold = lineas
                        else:
                            nueva_pieza = pieza_en_hold
                            if usar_estrategia_rapida:
                                punt_tmp, col_tmp, rot_tmp = self.mejor_placement_rapido(matriz_tablero, nueva_pieza, nivel)
                            else:
                                punt_tmp, col_tmp, rot_tmp = self.mejor_placement(matriz_tablero, nueva_pieza)
                            if col_tmp is not None:
                                nuevo_tablero, lineas = self.simular_placement(matriz_tablero, nueva_pieza, rot_tmp, col_tmp)
                                punt_hold = self.puntuar_tablero(nuevo_tablero, lineas)
                                hold_accion = 'swap'
                                nueva_pieza_hold = nueva_pieza
                                col_hold_obj = col_tmp
                                rot_hold_obj = rot_tmp
                                lineas_hold = lineas
                    if puede_colocar_actual and punt_actual >= punt_hold:
                        _, lineas_elim = self.simular_placement(matriz_tablero, pieza_actual, rot_obj, col_obj)
                        lineas_totales += lineas_elim
                        exito = self.colocar_pieza_mejorada(pieza_actual, columna_spawn, col_obj, rot_obj, keyboard, spawn_region, cell_w, nivel, usar_estrategia_rapida)
                        if exito:
                            ultimo_hold_usado = False
                        else:
                            contador_fallos += 1
                        pieza_actual = None
                    elif hold_accion is not None:
                        tiempos = self.obtener_tiempos_hold(nivel, usar_estrategia_rapida)
                        keyboard.press(Key.shift)
                        time.sleep(tiempos['hold_press'])
                        keyboard.release(Key.shift)
                        time.sleep(tiempos['hold_release_delay'])
                        time.sleep(0.03 if usar_estrategia_rapida else 0.05)
                        spawn_img_new = self.capturar_pantalla(spawn_region)
                        piezas_spawn_new = self.detectar_piezas_spawn(spawn_img_new)
                        if piezas_spawn_new:
                            nueva_x_spawn = piezas_spawn_new[0][0]
                            nueva_columna_spawn = int(round(nueva_x_spawn / cell_w))
                            nueva_columna_spawn = max(0, min(nueva_columna_spawn, self.BOARD_COLS - 1))
                            exito = self.colocar_pieza_mejorada(nueva_pieza_hold, nueva_columna_spawn, col_hold_obj, rot_hold_obj, keyboard, spawn_region, cell_w, nivel, usar_estrategia_rapida)
                            if exito:
                                ultimo_hold_usado = True
                                lineas_totales += lineas_hold
                            else:
                                contador_fallos += 1
                        else:
                            pieza_actual = nueva_pieza_hold
                        pieza_actual = None
                    else:
                        pieza_actual = None
                time.sleep(0.05 if nivel < 7 else 0.02)
        except KeyboardInterrupt:
            print("\nBot detenido")
        except Exception as e:
            print(f"Error en el bot: {e}")
            import traceback
            traceback.print_exc()

    # ----------------------- menu + helpers to run -----------------------
    def menu(self):
        while True:
            print("\nBOT TETRIO")
            print("1 - Calibrar zonas")
            print("2 - Ejecutar bot (modo adaptativo)")
            print("3 - Ver detección")
            print("4 - Calibrar colores SPAWN")
            print("5 - Calibrar colores NEXT/HOLD")
            print("6 - Salir")
            op = input("> ")
            if op == "1":
                self.modo_calibracion()
            elif op == "2":
                self.compute()
            elif op == "3":
                self.visualizar_zonas()
            elif op == "4":
                self.calibrar_colores_spawn()
            elif op == "5":
                self.calibrar_colores_next_hold()
            elif op == "6":
                break


if __name__ == "__main__":
        bot = bot_AzucarMorena()
        bot.cargar_colores_calibrados()
        bot.menu()
