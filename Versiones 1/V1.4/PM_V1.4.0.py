import pickle 
import cv2 
import mediapipe as mp 
import numpy as np 
import pyttsx3 as tts
import threading

model_dicc = pickle.load(open("Python\TR\Pickles\model.p", "rb"))
model = model_dicc["model"]

engine = tts.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-100)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

labels_dicc = { "A" : "A", "B" : "B",
                "C" : "C", "D" : "D", 
                "E" : "E", "F" : "F", 
                "I" : "I", "K" : "K", 
                "L" : "L", "M" : "M", 
                "N" : "N", "None" : " ", 
                "O" : "O", "P" : "P", 
                "Q" : "Q", "R" : "R", 
                "S" : "S", "T" : "T", 
                "U" : "U", "Y" : "Y"} 

romper = False

bloqueo = threading.Lock()

def main():
    global prediccion_1
    global caracter_predicho

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        H, W, _ = frame.shape 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(frame_rgb)
        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, mp_drawing_styles.get_default_hand_landmarks_style(), mp_drawing_styles.get_default_hand_connections_style() 
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
            
            prediccion_1 = model.predict(np.asarray([data_aux]))
            caracter_predicho = labels_dicc[str(prediccion_1[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
            cv2.putText(frame, caracter_predicho, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)
            if romper == True:
                break
        cv2.imshow("frame", frame)
        cv2.waitKey(1)


def extra():
    global frase, romper, prediccion_1
    frase = ""
    while True:
        if cv2.waitKey(27) == 27: 
            with bloqueo:
                global romper
                romper = True
                cv2.destroyAllWindows()
                break

        if cv2.waitKey(1) == ord("s"):
            if prediccion_1 != "None":
                frase += "".join(prediccion_1)
                print("Letra añadida a la oración")
            else: 
                frase += "".join(" ")
                print("Empezando siguiente palabra")  

if __name__ == "__main__":
    proceso1 = threading.Thread(target=main)
    proceso2 = threading.Thread(target=extra)

    proceso2.start()
    proceso1.start()

    proceso1.join()
    proceso2.join()
    print("Ejecutando programa")

    print(frase)
    engine.say(frase)
    engine.runAndWait()
