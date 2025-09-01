import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Ruta al modelo entrenado
model_path = 'vgg_transfer_model.h5'

# Define las clases de tu dataset en el mismo orden que las cargaste en el entrenamiento
# Si tenías 'gatos' y 'perros', usa ['gatos', 'perros']
class_names = ['class1', 'class2']  # <<--- REEMPLAZA CON LOS NOMBRES DE TUS CLASES REALES

# Define el tamaño de la imagen (el mismo que usaste para el entrenamiento)
IMG_SIZE = (224, 224)

# Carga el modelo guardado
try:
    model = load_model(model_path)
    print("Modelo cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Función para predecir una imagen
def predict_new_image(img_path):
    # Verifica si la imagen existe
    if not os.path.exists(img_path):
        print(f"Error: No se encontró la imagen en la ruta '{img_path}'.")
        return

    # Carga la imagen y la redimensiona al tamaño correcto
    img = image.load_img(img_path, target_size=IMG_SIZE)
    # Convierte la imagen a un array de NumPy
    img_array = image.img_to_array(img)
    # Agrega una dimensión de lote (batch) al array
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normaliza los valores de píxeles (importante si se hizo en el entrenamiento)
    # VGG16 se entrenó con un preprocesamiento específico, aunque image_dataset_from_directory
    # ya lo maneja. Es buena práctica asegurarte de que el preprocesamiento sea el mismo.
    
    # Realiza la predicción
    predictions = model.predict(img_array)
    
    # Obtiene el índice de la clase con la probabilidad más alta
    predicted_class_index = np.argmax(predictions[0])
    
    # Obtiene el nombre de la clase
    predicted_class = class_names[predicted_class_index]
    
    # Imprime el resultado
    print(f"\nLa imagen es predicha como: {predicted_class}")
    print(f"Probabilidades de predicción: {predictions[0]}")

# Ejemplo de uso: Reemplaza 'path/to/your/new_image.jpg' con la ruta de una imagen que quieras clasificar
# Por ejemplo, una imagen que no esté en tus directorios de entrenamiento o validación.
# predict_new_image('path/to/your/new_image.jpg')
# Ejemplo con una imagen de tu carpeta de validación
predict_new_image('/home/walter/Escritorio/Britney_Spears_(8514687036).jpg')