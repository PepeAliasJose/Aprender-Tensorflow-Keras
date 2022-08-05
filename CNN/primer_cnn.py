import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#Descargar datos
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255
test_images = test_images / 255


plt.figure()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Modelo # 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu, input_shape=(28,28,1)),
    # Capa convolucion que devuelve 32 imagenes con un filtro 3x3
    tf.keras.layers.MaxPooling2D((2,2), strides=2),
    # Redduccion de la imagen a la mitad con un 2x2 que coje el pixel con ams valor

    #tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    #tf.keras.layers.MaxPooling2D((2,2), strides=2),

    tf.keras.layers.Flatten(), # Aplana las imagenes de 2 dimensiones a 1 dimension 
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy']
)
historial = model.fit(train_images,train_labels, epochs=10)

plt.xlabel('Pasos')
plt.ylabel('Precision')
plt.plot(historial.history['accuracy'])
plt.title("Funcion precision")
plt.show()

#model.save('red_cnn/ropa_cnn.h5')

# model = tf.keras.models.load_model('red_cnn/ropa_cnn.h5')

# MOSTRAR RESULTADOS #
predictions = model.predict(test_images)
bien = 0
mal = 0
for x in range(10000):
    if(np.argmax(predictions[x]) == test_labels[x]):
        bien += 1
    else:
        mal += 1
    if(x%100 == 0):
        print(np.argmax(predictions[x]), " --- " , test_labels[x])

print("[BIEN]:{}".format(bien))
print("[MAL]:{}".format(mal))

'''
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
'''