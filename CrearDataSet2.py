import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Data2'

data = []
labels = []
max_num_points = 21

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

            if len(x_) >= max_num_points and len(y_) >= max_num_points:
                x_y_sorted = sorted(zip(x_, y_), key=lambda p: p[0])
                x_sorted, y_sorted = zip(*x_y_sorted)

                x_min, x_max = min(x_sorted), max(x_sorted)
                y_min, y_max = min(y_sorted), max(y_sorted)
                x_normalized = [(x - x_min) / (x_max - x_min) for x in x_sorted]
                y_normalized = [(y - y_min) / (y_max - y_min) for y in y_sorted]

                for i in range(max_num_points):
                    data_aux.append(x_normalized[i])
                    data_aux.append(y_normalized[i])

                data.append(data_aux)
                labels.append(dir_)

f = open('data4.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
