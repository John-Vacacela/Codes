import cv2
import os
import numpy as np
from keras.models import load_model


# Detectar y reconocer caras en tiempo real

def Reconocimiento_Facial():
    Path = 'FOTOS'
    imagePaths = os.listdir(Path)
    model = load_model('keras_model.h5')

    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if ret == False: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3,5)

        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h , x:x+w]
            rostro = cv2.resize(rostro, (720,720), interpolation=cv2.INTER_CUBIC)
            rostro = rostro.reshape(1,720,720,1)

            result = model.predict(rostro)
            # obtenemos el indice del valor mas alto
            indice_clase = np.argmax(result)
            #cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)
            #print(result)
            cv2.putText(frame, imagePaths[indice_clase], (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Reconocimiento_Facial()