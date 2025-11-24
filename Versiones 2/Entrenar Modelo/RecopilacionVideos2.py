import cv2
import os
import time

cap = cv2.VideoCapture(0)

desired_fps = 30
frame_interval = int(cap.get(5) / desired_fps) 

output_folder = 'videosTR1/'
numero_gestos = int(input("Cuantos gestos dinámicos quieres detectar en total? "))
videos_por_gesto = 20

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in numero_gestos:
    if not os.path.exists(os.path.join(output_folder, str(i))): 
        os.makedirs(os.path.join(output_folder, str(i))) 
    
    print("Recogiendo información para el gesto número: {}".format(i))

    for j in range(videos_por_gesto):
        output_video_path = f'{output_folder}output_video_{i}.mp4'

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = desired_fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval == 0:
                out.write(frame)

            frame_count += 1

            cv2.imshow('Video', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        out.release()
        print(f'Video {i} creado en {output_video_path}')
        time.sleep(1)

cv2.destroyAllWindows()
cap.release()