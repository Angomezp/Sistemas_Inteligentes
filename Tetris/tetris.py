import pyautogui
import cv2
import numpy as np
import time
from pynput import keyboard

############################################
# CONFIGURACION REGIONES PANTALLA
############################################

BOARD_REGION = (500,200,300,600)

NEXT1_REGION = (820,220,80,80)
NEXT2_REGION = (820,300,80,80)
NEXT3_REGION = (820,380,80,80)

############################################
# TECLAS CONTROL
############################################

LEFT="left"
RIGHT="right"
ROTATE="up"
DROP="space"

############################################
# VARIABLES GLOBALES
############################################

bot_running=False

PIECES=['I','O','T','S','Z','J','L']

############################################
# CARGAR SPRITES
############################################

def load_sprites():

    sprites={}

    for p in PIECES:

        img=cv2.imread(f"sprites/{p}.png")

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        _,binary=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

        sprites[p]=binary

    return sprites

############################################
# CAPTURA PANTALLA
############################################

def capture(region):

    shot=pyautogui.screenshot(region=region)

    frame=np.array(shot)

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,binary=cv2.threshold(gray,120,255,cv2.THRESH_BINARY)

    return binary,frame

############################################
# DETECTAR PIEZA
############################################

def detect_piece(region,sprites):

    binary,frame=capture(region)

    best_piece=None
    best_score=0

    for name,temp in sprites.items():

        res=cv2.matchTemplate(binary,temp,cv2.TM_CCOEFF_NORMED)

        _,score,_,_=cv2.minMaxLoc(res)

        if score>best_score:

            best_score=score
            best_piece=name

    return best_piece,frame

############################################
# DIBUJAR CENTRO
############################################

def draw_center(frame):

    h,w,_=frame.shape

    cx=w//2
    cy=h//2

    cv2.circle(frame,(cx,cy),5,(0,0,0),-1)

############################################
# CALIBRACION
############################################

def calibration_mode(sprites):

    while True:

        p1,f1=detect_piece(NEXT1_REGION,sprites)
        p2,f2=detect_piece(NEXT2_REGION,sprites)
        p3,f3=detect_piece(NEXT3_REGION,sprites)

        draw_center(f1)
        draw_center(f2)
        draw_center(f3)

        cv2.imshow("Next1",f1)
        cv2.imshow("Next2",f2)
        cv2.imshow("Next3",f3)

        if cv2.waitKey(1)==27:
            break

############################################
# DETECTAR TABLERO
############################################

def detect_board():

    binary,_=capture(BOARD_REGION)

    grid=np.zeros((20,10))

    h,w=binary.shape

    ch=h//20
    cw=w//10

    for r in range(20):

        for c in range(10):

            cell=binary[r*ch:(r+1)*ch,c*cw:(c+1)*cw]

            if np.mean(cell)>80:

                grid[r][c]=1

    return grid

############################################
# HEURISTICAS
############################################

def column_heights(board):

    heights=[]

    for c in range(10):

        h=0

        for r in range(20):

            if board[r][c]==1:

                h=20-r
                break

        heights.append(h)

    return heights

def holes(board):

    count=0

    for c in range(10):

        block=False

        for r in range(20):

            if board[r][c]==1:

                block=True

            elif block:

                count+=1

    return count

def bumpiness(heights):

    total=0

    for i in range(9):

        total+=abs(heights[i]-heights[i+1])

    return total

def lines(board):

    return sum([1 for r in board if sum(r)==10])

############################################
# COLUMNA RESERVADA
############################################

def reserve_penalty(board,piece):

    heights=column_heights(board)

    near_complete=sum(1 for r in board if sum(r)>=9)

    if near_complete>=5 and piece!='L':

        return 0

    return heights[9]*0.5

############################################
# EVALUACION
############################################

def evaluate(board,piece):

    heights=column_heights(board)

    score=0

    score+=lines(board)*3
    score-=holes(board)*7
    score-=bumpiness(heights)*0.6
    score-=sum(heights)*0.35
    score-=reserve_penalty(board,piece)

    return score

############################################
# SIMULACION
############################################

def simulate(board,piece,rot,x):

    new=np.copy(board)

    heights=column_heights(new)

    y=20-heights[x]-1

    if y<0:
        y=0

    if x<0 or x>9:
        return new

    new[y][x]=1

    return new

############################################
# BEAM SEARCH
############################################

def beam_search(board,pieces,depth=3,beam_width=8):

    states=[(board,0,None)]

    for d in range(depth):

        new_states=[]

        piece=pieces[d]

        for b,score,move in states:

            for rot in range(4):

                for x in range(10):

                    nb=simulate(b,piece,rot,x)

                    s=evaluate(nb,piece)

                    if d==0:
                        m=(rot,x)
                    else:
                        m=move

                    new_states.append((nb,score+s,m))

        new_states.sort(key=lambda x:x[1],reverse=True)

        states=new_states[:beam_width]

    return states[0][2]

############################################
# EJECUTAR MOVIMIENTO
############################################

def play_move(rot,x):

    spawn=4

    for _ in range(rot):
        pyautogui.press(ROTATE)
        time.sleep(0.01)

    dx=x-spawn

    if dx>0:
        for _ in range(dx):
            pyautogui.press(RIGHT)
            time.sleep(0.01)

    else:
        for _ in range(abs(dx)):
            pyautogui.press(LEFT)
            time.sleep(0.01)

    pyautogui.press(DROP)

############################################
# COLA PIEZAS
############################################

def init_next_pieces(sprites):

    p1,_=detect_piece(NEXT1_REGION,sprites)
    p2,_=detect_piece(NEXT2_REGION,sprites)
    p3,_=detect_piece(NEXT3_REGION,sprites)

    return [p1,p2,p3]

def update_next(queue,sprites):

    new,_=detect_piece(NEXT3_REGION,sprites)

    queue.append(new)

    if len(queue)>3:
        queue.pop(0)

    return queue

############################################
# BOT PRINCIPAL
############################################

def run_bot():

    sprites=load_sprites()

    next_pieces=init_next_pieces(sprites)

    while bot_running:

        if None in next_pieces:
            continue

        board=detect_board()

        move=beam_search(board,next_pieces)

        play_move(move[0],move[1])

        time.sleep(0.15)

        next_pieces.pop(0)

        next_pieces=update_next(next_pieces,sprites)

############################################
# MENU
############################################

def menu():

    sprites=load_sprites()

    while True:

        print("\n===== TETRIS BOT =====")
        print("1 iniciar bot (presiona B)")
        print("2 modo calibracion")
        print("3 salir")

        op=input("> ")

        if op=="1":
            print("esperando tecla B...")

        elif op=="2":
            calibration_mode(sprites)

        elif op=="3":
            break

############################################
# ACTIVAR BOT
############################################

def on_press(key):

    global bot_running

    try:

        if key.char=='b':

            print("BOT ACTIVADO")

            bot_running=True

            run_bot()

    except:
        pass

############################################
# LISTENER
############################################

listener=keyboard.Listener(on_press=on_press)

listener.start()

menu()