import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# --- 1. Cargar y Preprocesar el Conjunto de Datos ---


# Dataset FER2013 como ejemplo:
dataset, info = tfds.load("fer2013", as_supervised=True, with_info=True)

# Extraer imágenes y etiquetas
def preprocess_data(dataset):
    images, labels = [], []
    for image, label in dataset:
        image = tf.image.resize(image, (48, 48))  # Redimensionar a 48x48
        image = tf.image.rgb_to_grayscale(image)  # Convertir a escala de grises
        images.append(image.numpy())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

# Convertir el dataset a numpy
train_data, test_data = dataset['train'], dataset['test']
train_images, train_labels = preprocess_data(train_data)
test_images, test_labels = preprocess_data(test_data)

# Normalizar imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Dividir datos en entrenamiento y validación
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# --- 2. Aumentación de Datos ---
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen.fit(train_images)

# --- 3. Definir el Modelo CNN Mejorado ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(48, 48, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # Congelamos las capas del modelo preentrenado

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Dropout para regularización
    layers.Dense(8, activation='softmax')  # Ajustar según el número de emociones
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Resumen del modelo
model.summary()

# --- 4. Entrenamiento del Modelo ---
# Configurar callbacks para Early Stopping y TensorBoard
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=32),
    epochs=20,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping, tensorboard_callback]
)

# Evaluar el modelo en los datos de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# --- 5. Matriz de Confusión ---
predictions = model.predict(test_images)
y_pred = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, y_pred)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Verdadera')
plt.show()

# --- 6. Convertir el Modelo a TensorFlow Lite ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Cuantización para optimización
tflite_model = converter.convert()

# Guardar el modelo cuantizado en formato TensorFlow Lite
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo entrenado y convertido a TensorFlow Lite con cuantización")
