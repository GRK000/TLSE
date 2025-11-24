import os
import cv2

#DATA_DIR = './Data' 
DATA_DIR = "./BORRAR"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR) 

numero_de_subcarpetas = 20 
tamaño_dataset = 100 

cap = cv2.VideoCapture(0) 
for i in range(numero_de_subcarpetas): 
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i))) 
    
    print("Recogiendo información para la carpeta {}".format(i)) 

    while True: 
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) == ord("q"):
            break

    contador = 0 
    while contador < tamaño_dataset:
        ret, frame = cap.read() 
        cv2.imshow("frame", frame)
        cv2.waitKey(25) 
        cv2.imwrite(os.path.join(DATA_DIR, str(i), "{}.jpg".format(contador)), frame) 
        contador += 1 

cap.release()
cv2.destroyAllWindows() 