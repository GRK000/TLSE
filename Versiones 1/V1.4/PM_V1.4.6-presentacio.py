import cv2
import mediapipe as mp
import numpy as np
import pyttsx3 as tts
import pickle
import threading
from pynput import keyboard as kb
from os import system
import time

model_dicc = pickle.load(open("TR\Pickles\model.p", "rb"))
model = model_dicc["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.75)

labels_dicc = {
    "A": "A", "B": "B", "C": "C", "D": "D",
    "E": "E", "F": "F", "I": "I", "K": "K",
    "L": "L", "M": "M", "N": "N", "None": " ",
    "O": "O", "P": "P", "Q": "Q", "R": "R",
    "S": "S", "T": "T", "U": "U", "Y": "Y"
}

romper = False
prediccion_1 = ""
frase = ""
error_detectado = False

def esperar_tecla(tecla):
    global romper, frase
    if tecla == kb.KeyCode.from_char("s"):
        if prediccion_1 != "None":
            frase += "".join(prediccion_1)
            print("Lletra afegida a l'oració:", prediccion_1)
        else:
            frase += " "
            print("Començant seguent paraula")

    if tecla == kb.KeyCode.from_char('b'):
        frase = frase[:-1]
        print(f"Últim caràcter eliminat: {frase}")

    if tecla == kb.Key.esc:
        romper = True

    if tecla == kb.KeyCode.from_char('t'):
        frase = ""
        print(f"Tota la predicció eliminada")

    if tecla == kb.KeyCode.from_char("l"):
        engine = tts.init()

        rate = engine.getProperty('rate')
        engine.setProperty('rate', rate-100)

        print(frase)
        engine.say(frase)
        engine.runAndWait()
        engine.startLoop(False)
        engine = None

def main():
    global frase, prediccion_1, romper, error_detectado, reloj
    error_detectado = False 
    cap.set(3, 640)
    cap.set(4, 480)
    ptime = 0
    ctime = 0
    while not romper:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el fotograma de la càmera.")
                break

            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            data_aux = []
            x_ = []
            y_ = []

            resultado = hands.process(frame_rgb)
            if resultado.multi_hand_landmarks:
                for hand_landmarks in resultado.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style()
                        #frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                for hand_landmarks in resultado.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for j in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[j].x
                        y = hand_landmarks.landmark[j].y

                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 20
                y1 = int(min(y_) * H) - 20

                x2 = int(max(x_) * W) + 20
                y2 = int(max(y_) * H) + 20

                data_aux = np.array(data_aux).reshape(1, -1)

                prediccion_1 = model.predict(data_aux)
                caracter_predicho = labels_dicc[str(prediccion_1[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
                cv2.putText(frame, caracter_predicho, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
            cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)

            cv2.imshow("frame", frame)
            cv2.waitKey(1)

        except ValueError:
            if not error_detectado:
                print(f"\nError en la detecció de gestos: ValueError\nComproba que la càmera no detecti dues mans o més a la vegada")
                reloj = threading.Timer(5, espera)
                error_detectado = True
                reloj.start()

        if romper: 
            cap.release()
            cv2.destroyAllWindows()  
            print(frase)
            engine = tts.init()

            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate-100)

            engine.say(frase)
            engine.runAndWait()
            proceso1._stop()
            proceso2._stop()
            proceso3._stop()
            break
            
def escucha():
    while not romper:
        escuchador = kb.Listener(on_press=esperar_tecla, suppress=True)
        escuchador.start()
    
def espera(): 
    global error_detectado        
    error_detectado = False

if __name__ == "__main__":
    proceso1 = threading.Thread(target=main)
    proceso2 = threading.Thread(target=escucha)
    proceso3 = threading.Thread(target=esperar_tecla)

    proceso1.start()
    proceso2.start() 
    proceso3.start()

    system("cls")
    print("Tecles funcionals:\n S --> afegir lletra a la seqüència\n B --> Borrar l'última predicció guardada\n esc --> Reproduir seqüencia en veu alta y tancar el programa\n T --> Eliminar tota la seqüencia per tornar a començar\n")

    proceso1.join()
    proceso2.join()
    proceso3.join()

    cap.release()
    cv2.destroyAllWindows()
