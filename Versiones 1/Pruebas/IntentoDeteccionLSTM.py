import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

gesture_sequence = []
sequence_length = 30 
gesture_labels = {0: "Fist", 1: "L", 2: "Open Hand", 3: "Peace", 4: "Thumbs Up", 5: "None"}

model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, 42)))
model.add(Dropout(0.5))
model.add(Dense(len(gesture_labels), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Cargar el modelo entrenado previamente o entrenar uno nuevo con tus datos

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el fotograma a RGB y detectar manos
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        hand_data = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]).flatten()

        # Agregar el gesto al conjunto de datos
        gesture_sequence.append(hand_data)

        # Mantener el conjunto de datos a una longitud fija
        if len(gesture_sequence) > sequence_length:
            gesture_sequence.pop(0)

        # Cuando se ha capturado una secuencia completa, realizar la predicción
        if len(gesture_sequence) == sequence_length:
            sequence_data = np.expand_dims(np.array(gesture_sequence), axis=0)
            gesture_prediction = model.predict(sequence_data)

            # Obtener el gesto predicho
            predicted_gesture = gesture_labels[np.argmax(gesture_prediction)]

            # Mostrar el resultado en el fotograma
            cv2.putText(frame, predicted_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
