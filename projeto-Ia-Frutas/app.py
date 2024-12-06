import tensorflow as tf
from tensorflow.keras.layers import TFSMLayer
from pathlib import Path
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Classes ajustadas
classes = [
    "Maçã", "Banana", "Laranja", "Abacate", "Coco", 
    "Milho", "Pera", "Melancia", "Maracuja", "Manga"
]

def check_model_path(saved_model_dir):
    if not os.path.exists(saved_model_dir):
        return False
    model_file_path = os.path.join(saved_model_dir, 'saved_model.pb')
    if not os.path.exists(model_file_path):
        return False
    return True

# Caminho do modelo original e do modelo convertido
SAVED_MODEL_DIR = r"C:\Users\gugab\Documents\faculdade\5o Periodo\TAV\projeto-Ia-Frutas\converted_savedmodel\model.savedmodel"
H5_MODEL_PATH = r"C:\Users\gugab\Documents\faculdade\5o Periodo\TAV\projeto-Ia-Frutas\Frutas_base\meu_modelo.keras"

def convert_to_h5(saved_model_dir, h5_model_path):
    try:
        model = tf.saved_model.load(saved_model_dir)
        model.save(h5_model_path)
    except Exception as e:
        print(f"Erro ao converter o modelo SavedModel: {e}")

def load_teachable_machine_model(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None
    return model

# Função para fazer uma previsão em uma imagem
def predict(model, img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar para 224x224
        img_array = image.img_to_array(img)  # Converter para array NumPy
        img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão do batch
        img_array = img_array / 255.0  # Normalizar para o intervalo [0, 1]

        # Realizar a previsão
        predictions = model.predict(img_array)

        # Verificar se a previsão é um dicionário e acessar a chave correta
        if isinstance(predictions, dict):
            predictions = predictions['sequential_7']  # A chave 'sequential_7' contém as previsões

        # Garantir que predictions seja um array NumPy
        if isinstance(predictions, np.ndarray):
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_class_name = classes[predicted_class]
        else:
            raise ValueError("O formato de saída do modelo não é esperado.")

    except Exception as e:
        print(f"Erro ao fazer a previsão: {e}")
        return None

    return predicted_class_name

def main():
    if not check_model_path(SAVED_MODEL_DIR):
        return
    try:
        if not Path(H5_MODEL_PATH).exists():
            convert_to_h5(SAVED_MODEL_DIR, H5_MODEL_PATH)

        model = load_teachable_machine_model(H5_MODEL_PATH)
        if model is None:
            return

        img_path = Path(r"C:\Users\gugab\Documents\faculdade\5o Periodo\TAV\projeto-Ia-Frutas\uploaded_images\maca.jpg")
        prediction = predict(model, img_path)

        if prediction:
            print(f"A imagem '{img_path.name}' foi classificada como: {prediction}")
        else:
            print("Não foi possível fazer a previsão.")

    except Exception as e:
        print(f"Ocorreu um erro no processo principal: {e}")

if __name__ == "__main__":
    main()
