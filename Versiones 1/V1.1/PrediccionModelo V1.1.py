import pickle 
import cv2 
import mediapipe as mp 
import numpy as np 
import time

model_dicc = pickle.load(open("Python\TR\Pickles\model.p", "rb"))
model = model_dicc["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dicc = { "A" : "A", "B" : "B",
                "C" : "C", "D" : "D", 
                "E" : "E", "F" : "F", 
                "I" : "I", "K" : "K", 
                "L" : "L", "M" : "M", 
                "N" : "N", "O" : "O",
                "P" : "P", "Q" : "Q", 
                "R" : "R", "S" : "S", 
                "T" : "T", "U" : "U", 
                "Y" : "Y", "None" : "" } 

while True: 
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape 
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    resultado = hands.process(frame_rgb)
    if cv2.waitKey(25)%256 == 27:
        print("Esc pulsado, cerrando programa...")
        break

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

        x1 = int(min(x_) * W) - 45
        y1 = int(min(y_) * H) - 45

        x2 = int(max(x_) * W) + 20
        y2 = int(max(y_) * H) + 20

        prediccion = model.predict([np.asarray(data_aux)])
        caracter_predicho = labels_dicc[str(prediccion[0])]


        ctime = time.time()
        ptime = ctime + 0.0001  
        fps = 1/(ctime-ptime)
        

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3) 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 3)
        cv2.putText(frame, caracter_predicho, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,0), 3, cv2.LINE_AA)

    cv2.imshow("Traductor de lengua de signos", frame)
    cv2.waitKey(1)

    if cv2.waitKey(25)%256 == 27: 
        print("Esc pulsado, cerrando programa...")
        break