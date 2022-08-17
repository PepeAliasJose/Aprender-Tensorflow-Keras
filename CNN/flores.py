import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import glob
import shutil

# Descargar las imagenes 
url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
dir = tf.keras.utils.get_file('flower_photos.tgz', origin=url, extract = True)
base = os.path.join(os.path.dirname(dir), 'flower_photos')

# Nombre de las clases
clases = ['roses','daisy','dandelion','sunflowers','tulips']

'''
# bucle que mueve las imagenes a una estructura de carpetas
for cl in clases:
    img_path = os.path.join(base,cl)
    imagenes = glob.glob(img_path+'/*.jpg')
    print("{}:{} Images".format(cl, len(imagenes)))
    train, val = imagenes[:round(len(imagenes)*0.8)], imagenes[round(len(imagenes)*0.8):]

    for t in train:
        if not os.path.exists(os.path.join(base,'train',cl)):
            os.makedirs(os.path.join(base,'train',cl))
        shutil.move(t,os.path.join(base,'train',cl))
    
    for v in val:
        if not os.path.exists(os.path.join(base,'validar',cl)):
            os.makedirs(os.path.join(base,'validar',cl))
        shutil.move(v,os.path.join(base,'validar',cl))
'''

train_dir = os.path.join(base,'train')
valida_dir = os.path.join(base,'validar')

BATCH = 100
SHAPE = 150

# Ajusta el generador de imagenes con rotacion reescalados etc
train_image_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=48,
                                            width_shift_range=0.3,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            fill_mode='nearest')
# Genera las imagenes con el filtro anterior por cada una y las guarda en una variable
train_data = train_image_generator.flow_from_directory(batch_size=BATCH,
                                                        directory = train_dir, 
                                                        shuffle=True, 
                                                        target_size=(SHAPE,SHAPE), 
                                                        class_mode='sparse')


val_image_generator = ImageDataGenerator(rescale=1./255)

validation_data = val_image_generator.flow_from_directory(batch_size=BATCH,
                                                    directory = valida_dir, 
                                                    target_size=(SHAPE,SHAPE), 
                                                    class_mode='sparse')


modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(SHAPE,SHAPE,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

modelo.summary()

PASOS = 100

ENTRENAR = 2935
VALIDAR = 735

history = modelo.fit(train_data,
    steps_per_epoch=int(np.ceil(ENTRENAR/BATCH)),
    epochs = PASOS,
    validation_data = validation_data,
    validation_steps=int(np.ceil(VALIDAR/BATCH)),)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(PASOS)

modelo.save('red_cnn/flores_cnn.h5')

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epoch_range,acc,label='PRECISION')
plt.plot(epoch_range, val_acc, label='PRUEBA PRECISION')
plt.legend(loc='lower right')
plt.title('ENTRENAMIENTO Y PRUEBA DE PRECISON')

plt.subplot(1,2,2)
plt.plot(epoch_range,loss,label='PERDIDA')
plt.plot(epoch_range, val_loss, label='PRUEBA PERDIDA')
plt.legend(loc='upper right')
plt.title('ENTRENAMIENTO Y PRUEBA DE PERDIDA')
plt.show()