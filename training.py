import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16

# ---------------------------
# Configuración GPU
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ---------------------------
# Parámetros
# ---------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10
NUM_CAPTURES = 300
MODEL_PATH = "vgg_transfer_model.keras"

# ---------------------------
# 1. Pedir documento
# ---------------------------
doc_id = input("📄 Ingrese su número de documento: ")
save_dir = os.path.join("data", "train", doc_id)
os.makedirs(save_dir, exist_ok=True)
print(f"📂 Capturas se guardarán en: {save_dir}")
print("📸 Presione 'I' para empezar la captura (se tomarán 300 fotos).")

# ---------------------------
# 2. Captura de rostro
# ---------------------------
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = len(os.listdir(save_dir))
capturing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if capturing and count < NUM_CAPTURES:
            face = cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE)
            img_name = f"{doc_id}_{count}.jpg"
            cv2.imwrite(os.path.join(save_dir, img_name), face)
            count += 1
            cv2.putText(frame, f"Capturando {count}/{NUM_CAPTURES}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12), 2)

    cv2.imshow("Captura de Rostros", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("i"):
        capturing = True
        print("▶ Iniciando captura...")
    elif key == ord("q"):
        break

    if capturing and count >= NUM_CAPTURES:
        print("✅ Captura completada.")
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------
# 3. Crear datasets
# ---------------------------
train_dir = "data/train"

dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="both",
    seed=123
)
train_dataset, validation_dataset = dataset

num_classes = len(train_dataset.class_names)
print("Clases detectadas:", train_dataset.class_names)

# ---------------------------
# 4. Cargar o crear modelo
# ---------------------------
if os.path.exists(MODEL_PATH):
    print("⚡ Cargando modelo existente...")
    old_model = load_model(MODEL_PATH)

    base_model = old_model.layers[0]
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
else:
    print("⚡ Creando modelo nuevo...")
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------------------
# 5. Entrenamiento
# ---------------------------
print("⚡ Iniciando entrenamiento...")
history = model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)

# ---------------------------
# 6. Guardar modelo y clases
# ---------------------------
model.save(MODEL_PATH)
with open("class_names.txt", "w") as f:
    for cname in train_dataset.class_names:
        f.write(cname + "\n")

print("✅ Entrenamiento completado y modelo guardado.")
