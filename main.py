import os
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModel
import torch.nn as nn

class CustomCLIPModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CustomCLIPModel, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values=pixel_values)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return logits

path = os.getcwd()
# Definir las etiquetas
LABELS = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation',
          'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy',
          'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming',
          'selective_logging', 'blow_down']

# Definir el procesador y cargar el modelo entrenado
CLIP_PATH = os.path.join(path, 'model/clip-vit-base-patch16')
processor = AutoProcessor.from_pretrained(CLIP_PATH)
base_model = CLIPVisionModel.from_pretrained(CLIP_PATH)

# Inicializar el modelo
device = "cuda:0" if torch.cuda.is_available() else "cpu"
num_labels = len(LABELS)
model = CustomCLIPModel(base_model, num_labels)
model.to(device)

# Cargar el modelo entrenado
MODEL_PATH = os.path.join(path, 'model/CLIPclassification_1e.pth')
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Función de inferencia
def infer_image(image_path):
    # Cargar y procesar la imagen
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Redimensionar imagen
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # Realizar la inferencia
    with torch.no_grad():
        logits = model(pixel_values=pixel_values)

    # Aplicar sigmoid para convertir logits en probabilidades
    probs = torch.sigmoid(logits).cpu().numpy().flatten()

    # Crear un diccionario con las etiquetas y sus probabilidades
    results = {LABELS[i]: probs[i] for i in range(len(LABELS))}

    return results


def main():
    img_path = os.path.join(path, 'img/train_1.jpg')
    results = infer_image(img_path)
    print("Resultados de la inferencia:")
    for label, prob in results.items():
        print(f"{label}: {prob:.4f}")
    return(label, prob)

if __name__ == "__main__":
    main()