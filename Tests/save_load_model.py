import os
import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define un modelo secuencial simple
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()


checkpoint_path = "modelo_guardado_prueba/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Crea una llamada para guardar los pesos del modelo
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Puedes usar los pesos para otro modelo si los dos tienen la misma arquitectura

print("------- Entrenar modelo --------")
# Entrena el modelo
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Le pasas la llamada
# El callback guarda el modelo en la ruta especificada en "Checkpoint_path"

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.



# Crea un modelo sin entrenar
model2 = create_model()

print("Modelo nuevo:")
# Se mira la precision del modelo
loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


# Recarga los pesos guardados de antes en el modelo nuemo
model2.load_weights(checkpoint_path)
print("Modelo restaurado:")
# Se vuelve a evaluar el modelo restaurado
loss, acc = model2.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


print("Guardar modelo:")
#Guardar el modelo
model.save('saved_model/my_model.h5')
new_model = tf.keras.models.load_model('saved_model/my_model.h5')

print("Modelo cargado desde archivo:")
# Check its architecture
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))


