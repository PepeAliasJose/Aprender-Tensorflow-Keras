# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras

# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Descargar datos
fashion_mnist = keras.datasets.fashion_mnist
#Carga los datos de las imagenes en las variables
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#clases
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Configurar las capas
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#Flatten aplana las imagenes a una sola linea de pixeles
    keras.layers.Dense(128, activation='relu'),#Capa de 128 neuronas que procesa la informacion(relu)
    keras.layers.Dense(128, activation='relu'),#Capa de 128 neuronas que procesa la informacion(relu)
    keras.layers.Dense(10, activation='softmax') #Capa de 10 nodos que devuelve las 10 posibilidades de cada categoria(sofmax)
])

#Configuracion del modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Entrena el modelo
model.fit(train_images, train_labels, epochs=15)
print("/n MODELO: /n")
model.summary()
model.save('saved_model/ropa.h5')
