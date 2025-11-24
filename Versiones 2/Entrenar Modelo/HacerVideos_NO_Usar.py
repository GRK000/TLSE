import os
import cv2

input_folder = 'carpeta_de_videos'

output_folder = 'carpeta_de_videos_ajustados'

target_length = 100

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

video_files = [f for f in os.listdir(input_folder) if f.endswith('.mp4')]

for video_file in video_files:
    input_path = os.path.join(input_folder, video_file)

    output_path = os.path.join(output_folder, f'{video_file[:-4]}_ajustado.mp4')

    cap = cv2.VideoCapture(input_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    scale_factor = frame_count / target_length

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (640, 480))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for _ in range(int(scale_factor)):
            out.write(frame)

    cap.release()
    out.release()

    print(f'Video ajustado guardado en: {output_path}')

print('Proceso completado.')
