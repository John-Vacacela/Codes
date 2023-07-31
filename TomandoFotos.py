import cv2
import os

if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    # Definimos el clasificador de rostros
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    facedetect_profile = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    count = 0

    nameID = str(input('Ingresa tu nombre: ')).lower()
    path = 'FOTOS/'+nameID

    print('Creado')
    os.makedirs(path)

    while True:
        ret, frame = video.read()
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        faces_profile = facedetect_profile.detectMultiScale(gray,1.3,5)
        for x,y,w,h in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
            face = auxFrame[y:y+h,x:x+w]
            face = cv2.resize(face,(720,720), interpolation=cv2.INTER_CUBIC)
            name = './FOTOS/'+nameID+'/'+str(count)+'.jpg'
            cv2.imwrite(name,face)
            count += 1

        cv2.imshow('Facial Recognition', frame)
        cv2.waitKey(1)

        if count > 400:
            break

    video.release()
    cv2.destroyAllWindows()