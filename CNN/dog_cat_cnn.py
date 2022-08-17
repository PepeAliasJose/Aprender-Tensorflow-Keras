import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import matplotlib.pyplot as plt
import numpy as np
import logging

# Descargar las imagenes 
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
dir = tf.keras.utils.get_file('cat_and_dogs_filtered.zip', origin=url, extract = True)
zip_path = os.path.dirname(dir)
print(zip_path)

# variables con las rutas a las carpetas con imagenes
base = os.path.join(os.path.dirname(dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base,'train')
test_dir = os.path.join(base,'validation')

train_cat = os.path.join(train_dir,'cats')
train_dog = os.path.join(train_dir,'dogs')
test_cat = os.path.join(test_dir,'cats')
test_dog = os.path.join(test_dir,'dogs')

BATCH = 50
SHAPE = 150

### Cambiar los datos a un tipo adecuado para la red ###

#Generadores
train_image_generator = ImageDataGenerator(rescale=1./255,
                                            rotation_range=48,
                                            width_shift_range=0.3,
                                            height_shift_range=0.2,
                                            shear_range=0.2,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            fill_mode='nearest')

test_image_generator = ImageDataGenerator(rescale=1./255)

train_data = train_image_generator.flow_from_directory(batch_size=BATCH,
                                                        directory = train_dir, 
                                                        shuffle=True, 
                                                        target_size=(SHAPE,SHAPE), 
                                                        class_mode='binary')

test_data = test_image_generator.flow_from_directory(batch_size=BATCH,
                                                    directory = test_dir, 
                                                    shuffle=True, 
                                                    target_size=(SHAPE,SHAPE), 
                                                    class_mode='binary')


sample = [train_data[0][0][0] for i in range(5)]

def plot(arg):
    fig,axes = plt.subplots(1,5,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(arg,axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

plot(sample)

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(SHAPE,SHAPE,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(2)
])

modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

modelo.summary()

PASOS = 15

history = modelo.fit(
    train_data,
    steps_per_epoch=40,
    epochs = PASOS,
    validation_data = test_data,
    validation_steps=20
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(PASOS)

modelo.save('red_cnn/perro_gato_cnn.h5')

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