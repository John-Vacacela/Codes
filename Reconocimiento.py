import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

#CREACIÓN DEL MODELO

def cargar_imagenes(path):
    lista_personas = os.listdir(path)

    labels = []
    facesData = []
    label = 0

    for nameDir in lista_personas:
        personPath = path + '/' + nameDir
        print('Leyendo las imágenes', nameDir)

        for fileName in os.listdir(personPath):
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName, 0))
            image = cv2.imread(personPath+'/'+fileName, 0)
            #cv2.imshow('image', image)
            #cv2.waitKey(10)
        label = label + 1
    #cv2.destroyAllWindows()
    return facesData, labels, lista_personas

def entrenar_modelo(facesData, labels):
    facesData = np.array(facesData)
    labels = np.array(labels)

    #labelEncoder = LabelEncoder()
    #labels = labelEncoder.fit_transform(labels)
    labels = to_categorical(labels)

    # Crear el modelo
    model = Sequential()
    model.add(Flatten(input_shape=(facesData.shape[1],facesData.shape[2])))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(facesData, labels, epochs=15)

    return model

if __name__ == '__main__':
    path = 'FOTOS'
    facesData, labels, lista_personas = cargar_imagenes(path)
    model= entrenar_modelo(facesData, labels)
    model.save('keras_model.h5')  # Guardar el modelo entrenado
    #np.save('labelEncoder.npy', labelEncoder.classes_)  # Guardar el label encoder