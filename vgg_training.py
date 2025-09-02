import cv2
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ---------------------------
# Configuración GPU y Constantes
# ---------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10
NUM_CAPTURES = 100
MODEL_PATH = "vgg_transfer_model.keras"

# ---------------------------
# Funciones
# ---------------------------
def capture_faces(doc_id):
    """
    Captura fotos de un rostro usando la cámara y las guarda en un directorio.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return False

    save_dir = os.path.join("data", "train", doc_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Capturas se guardarán en: {save_dir}")
    print("Presione 'i' para empezar la captura. Presione 'q' para salir.")

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
            print("Iniciando captura...")
        elif key == ord("q"):
            break
        if capturing and count >= NUM_CAPTURES:
            print("Captura completada.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def create_datasets(train_dir="data/train"):
    """
    Crea y devuelve los datasets de entrenamiento y validación.
    """
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        print(f"Error: No se encontraron datos en el directorio '{train_dir}'.")
        return None, None

    dataset = image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset="both",
        seed=123
    )
    return dataset

def get_model(num_classes):
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo existente...")
        old_model = load_model(MODEL_PATH)

        old_classes = old_model.output_shape[-1]
        if old_classes == num_classes:
            print("El número de clases no cambió. Reusando el modelo completo.")
            model = old_model
        else:
            print(f"Ajustando la última capa: {old_classes} -> {num_classes}")
            # Quitamos la última capa y creamos una nueva salida
            x = old_model.layers[-2].output
            new_output = Dense(num_classes, activation="softmax", name=f"dense_out_{num_classes}")(x)
            model = tf.keras.Model(inputs=old_model.input, outputs=new_output)
    else:
        print("Creando modelo nuevo...")
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base_model.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(256, activation="relu", name="dense_hidden")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax", name=f"dense_out_{num_classes}")(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Fine-tuning
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model(model, train_dataset, validation_dataset):
    """
    Entrena el modelo y lo guarda.
    """
    print("Iniciando entrenamiento...")
    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)
    model.save(MODEL_PATH)
    with open("class_names.txt", "w") as f:
        for cname in train_dataset.class_names:
            f.write(cname + "\n")
    print("Entrenamiento completado y modelo guardado.\n")

def run_admin():
    """
    Función que maneja la lógica del modo admin.
    """
    print("\n--- Modo ADMIN ---")
    doc_id = input("Ingrese su número de documento: ")
    
    if capture_faces(doc_id):
        train_dataset, validation_dataset = create_datasets()
        if train_dataset and validation_dataset:
            num_classes = len(train_dataset.class_names)
            model = get_model(num_classes)
            train_model(model, train_dataset, validation_dataset)

# ---------------------------
# Ejecución directa en modo ADMIN
# ---------------------------
if __name__ == "__main__":
    run_admin()
