import tensorflow as tf
import numpy as np

#celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 42, 234, 4]) # Datos de temperatura en celsius
#fahre = np.array([-40, 14, 32, 46, 59, 72, 100, 107.6, 453.2, 39.2]) # La misma temperatura pero en fahrenheit

celsius = np.array([-40, -10, 0, 8, 15, 22, 38, 42, 234, 4, 12, 45, 675, 234, 123, 876, 34, 78, 94, 3, 56, 79, 43, 17, -43, 34])
fahre = ((celsius * (9/5))+32)

print("Celsius: ")
print(celsius)
print("Fahrenheit correcto:")
print(fahre)

""""
capa0 = tf.keras.layers.Dense(units=12, input_shape=[1]) # Una capa de 12 "neurona"
capa1 = tf.keras.layers.Dense(units=4) # Una capa de 4 "neuronas"
capa2 = tf.keras.layers.Dense(units=4) # Una capa de 4 "neuronas"
capa3 = tf.keras.layers.Dense(units=1) # Una capa de 1 "neurona"

modelo = tf.keras.Sequential([
    capa0,
    capa1,
    capa2,
    capa3
    ])

modelo.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.1),
                metrics=['accuracy'])

historial = modelo.fit(celsius,fahre, epochs=600, verbose=False)
# Se entrena el modelo|entrada|salida|pasos

import matplotlib.pyplot as plt
plt.xlabel('Pasos')
plt.ylabel('Perdida')
plt.plot(historial.history['loss'])
plt.title("Funcion perdida")


print("Variables de la capa: {}".format(capa0.get_weights()))
"""
modelo = tf.keras.models.load_model('redes/cel-fahre-complejo.h5')

print("Fahrenheit por la red: ")
print(modelo.predict(celsius))

#modelo.save('redes/cel-fahre-complejo.h5')
#plt.show()