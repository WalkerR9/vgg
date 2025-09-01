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
NUM_CAPTURES = 300
MODEL_PATH = "vgg_transfer_model.keras"
CLASS_NAMES_PATH = "class_names.txt"

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
    """
    Carga o crea un modelo VGG16 y lo prepara para el entrenamiento.
    """
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo existente...")
        model = load_model(MODEL_PATH)
    else:
        print("Creando modelo nuevo...")
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
    return model

def train_model(model, train_dataset, validation_dataset):
    """
    Entrena el modelo y lo guarda.
    """
    print("Iniciando entrenamiento...")
    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)
    model.save(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        for cname in train_dataset.class_names:
            f.write(cname + "\n")
    print("Entrenamiento completado y modelo guardado.\n")

def run_visor(model, class_names):
    """
    Ejecuta el modo de visualización y predicción en tiempo real.
    Permite cambiar al modo admin con la tecla 'a'.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return 'quit'

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    print("\n--- Modo VISOR ---")
    print("Presione 'a' para ir a modo ADMIN o 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE)
            face_array = np.expand_dims(face, axis=0) / 255.0
            
            predictions = model.predict(face_array, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            
            if confidence > 0.5:
                label = class_names[class_idx]
                color = (0, 255, 0)
            else:
                label = "Desconocido"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Predicción", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return 'quit'
        elif key == ord('a'):
            cap.release()
            cv2.destroyAllWindows()
            return 'admin'
    
    cap.release()
    cv2.destroyAllWindows()
    return None

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
# Bucle principal de control
# ---------------------------
current_mode = 'visor'
while True:
    if current_mode == 'visor':
        if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
            print("Error: No hay un modelo entrenado. Vaya al modo ADMIN para crearlo.\n")
            current_mode = 'admin'
            continue
            
        try:
            model = load_model(MODEL_PATH)
            with open(CLASS_NAMES_PATH) as f:
                class_names = [line.strip() for line in f]
            
            result = run_visor(model, class_names)
            if result == 'quit':
                break
            elif result == 'admin':
                current_mode = 'admin'
        except Exception as e:
            print(f"Ocurrió un error: {e}. Volviendo a modo admin.\n")
            current_mode = 'admin'

    elif current_mode == 'admin':
        run_admin()
        current_mode = 'visor'
        print("Finalizado modo ADMIN. Volviendo a modo VISOR.")