import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16

# ---------------------------
# Parámetros
# ---------------------------
MODEL_PATH = "vgg_transfer_model.keras"
train_dir = "data/train"
validation_dir = "data/validation"
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

# ---------------------------
# 1. Entrenar modelo si no existe
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("⚡ No se encontró el modelo, entrenando desde cero...")

    # Cargar datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    num_classes = len(train_dataset.class_names)
    print("Clases detectadas:", train_dataset.class_names)

    # Base VGG16
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    # Modelo final
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

    # Guardar
    model.save(MODEL_PATH)

    # Guardar clases en un archivo txt
    with open("class_names.txt", "w") as f:
        for cname in train_dataset.class_names:
            f.write(cname + "\n")

else:
    print("✅ Modelo encontrado, cargando...")

# ---------------------------
# 2. Cargar modelo y clases
# ---------------------------
model = load_model(MODEL_PATH)

if os.path.exists("class_names.txt"):
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    # fallback: si no existe, solo muestra índices
    class_names = [f"Clase {i}" for i in range(model.output_shape[-1])]

print("Clases:", class_names)

# ---------------------------
# 3. Reconocimiento en tiempo real
# ---------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, IMG_SIZE)
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        preds = model.predict(face)
        class_id = np.argmax(preds)
        confidence = np.max(preds)
        confidence_pct = confidence * 100

        # Definir etiqueta y color según confianza
        if confidence_pct >= 80:
            label = f"{class_names[class_id]}: {confidence_pct:.1f}%"
            color = (36, 255, 12)  # Verde
        else:
            label = "Desconocido"
            color = (0, 0, 255)  # Rojo

        # Dibujar rectángulo
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (36,255,12), 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
