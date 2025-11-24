import cv2
import os

def grabar_video(carpeta_salida, nombre_archivo, fps, duracion_segundos, resolucion=(640, 480)):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    ruta_completa = os.path.join(carpeta_salida, nombre_archivo)

    # Configurar la captura de video desde la cámara (puedes ajustar el índice de la cámara según tu configuración)
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(ruta_completa, fourcc, fps, resolucion)

    tiempo_inicio = cv2.getTickCount()

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        out.write(frame)

        tiempo_actual = cv2.getTickCount()
        tiempo_transcurrido = (tiempo_actual - tiempo_inicio) / cv2.getTickFrequency()

        if tiempo_transcurrido >= duracion_segundos:
            break

    cap.release()
    out.release()

if __name__ == "__main__":
    carpeta_salida = "videos_salida"
    nombre_archivo = "video_salida.avi"
    fps_deseados = 30
    duracion_segundos = 10

    grabar_video(carpeta_salida, nombre_archivo, fps_deseados, duracion_segundos)
