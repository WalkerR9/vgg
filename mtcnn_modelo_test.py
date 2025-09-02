import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from mtcnn import MTCNN

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

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

    model.save(MODEL_PATH)

    with open("class_names.txt", "w") as f:
        for cname in train_dataset.class_names:
            f.write(cname + "\n")
else:
    print("Modelo encontrado, cargando...")

# ---------------------------
# 2. Cargar modelo y clases
# ---------------------------
model = load_model(MODEL_PATH)

if os.path.exists("class_names.txt"):
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    class_names = [f"Clase {i}" for i in range(model.output_shape[-1])]

print("Clases:", class_names)

# ---------------------------
# 3. Reconocimiento en tiempo real con MTCNN
# ---------------------------
cap = cv2.VideoCapture(0)
detector = MTCNN()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir resolución para detección rápida
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_small)

    for face in faces:
        x, y, w, h = face['box']
        # Escalar coordenadas a tamaño original
        x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
        x, y = max(0, x), max(0, y)

        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, IMG_SIZE)
        face_img = face_img.astype("float32") / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        preds = model.predict(face_img)
        class_id = np.argmax(preds)
        confidence = np.max(preds)
        confidence_pct = confidence * 100

        # Etiqueta y color según confianza
        if confidence_pct >= 50:
            label = f"{class_names[class_id]}: {confidence_pct:.1f}%"
            color = (36, 255, 12)  # Verde
        else:
            label = "Desconocido"
            color = (0, 0, 255)  # Rojo

        # Dibujar rectángulo y texto
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, color, 2)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
