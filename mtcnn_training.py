import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from mtcnn import MTCNN

# ---------------------------
# Configuración GPU y constantes
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
# Función para capturar caras
# ---------------------------
def capture_faces(doc_id):
    """
    Captura fotos usando MTCNN para diferentes ángulos.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: no se pudo abrir la cámara.")
        return False

    save_dir = os.path.join("data", "train", doc_id)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Las capturas se guardarán en: {save_dir}")
    print("Presione 'i' para iniciar la captura y 'q' para salir.")

    detector = MTCNN()
    count = len(os.listdir(save_dir))
    capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if capturing and count < NUM_CAPTURES:
                face_img = cv2.resize(frame[y:y+h, x:x+w], IMG_SIZE)
                img_name = f"{doc_id}_{count}.jpg"
                cv2.imwrite(os.path.join(save_dir, img_name), face_img)
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

# ---------------------------
# Crear datasets con data augmentation
# ---------------------------
def create_datasets(train_dir="data/train"):
    datagen = ImageDataGenerator(
        rotation_range=30,        # simula diferentes ángulos
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.7,1.3],
        horizontal_flip=True,
        validation_split=0.2
    )

    train_dataset = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training"
    )

    validation_dataset = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation"
    )

    return train_dataset, validation_dataset

# ---------------------------
# Crear o ajustar modelo
# ---------------------------
def get_model(num_classes):
    if os.path.exists(MODEL_PATH):
        print("Cargando modelo existente...")
        old_model = load_model(MODEL_PATH)
        old_classes = old_model.output_shape[-1]

        if old_classes == num_classes:
            print("Reusando el modelo completo.")
            model = old_model
        else:
            print(f"Ajustando última capa: {old_classes} -> {num_classes}")
            x = old_model.layers[-2].output
            new_output = Dense(num_classes, activation="softmax", name=f"dense_out_{num_classes}")(x)
            model = tf.keras.Model(inputs=old_model.input, outputs=new_output)
    else:
        print("Creando modelo nuevo...")
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
        base_model.trainable = False
        x = Flatten()(base_model.output)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)

    # Fine-tuning: todas las capas entrenables
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ---------------------------
# Entrenar modelo
# ---------------------------
def train_model(model, train_dataset, validation_dataset):
    print("Iniciando entrenamiento...")
    model.fit(train_dataset, epochs=EPOCHS, validation_data=validation_dataset)
    model.save(MODEL_PATH)
    with open("class_names.txt", "w") as f:
        for cname in train_dataset.class_indices.keys():
            f.write(cname + "\n")
    print("Entrenamiento completado y modelo guardado.\n")

# ---------------------------
# Modo ADMIN
# ---------------------------
def run_admin():
    print("\n--- Modo ADMIN ---")
    doc_id = input("Ingrese su número de documento: ")
    if capture_faces(doc_id):
        train_dataset, validation_dataset = create_datasets()
        num_classes = len(train_dataset.class_indices)
        model = get_model(num_classes)
        train_model(model, train_dataset, validation_dataset)

# ---------------------------
# Ejecutar
# ---------------------------
if __name__ == "__main__":
    run_admin()
