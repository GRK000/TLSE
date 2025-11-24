import cv2
import mediapipe as mp
import numpy as np
import pickle
import os 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.75)

DATA_DIR = "./"

archivos = os.listdir(DATA_DIR)

videos = [archivo for archivo in archivos if archivo.endswith(".mp4")]

data = []
labels = []
for dir_ in videos:
    for video in dir_:
        video_path = os.path.join(DATA_DIR, dir_, video)
        for fotograma in video: 
            

            data_aux = []
            x_ = []
            y_ = []
            img = cv2.imread(fotograma)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            
    

f = open("data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()