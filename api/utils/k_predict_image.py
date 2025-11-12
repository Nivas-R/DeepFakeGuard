# api/utils/k_predict_image.py
import torch
from PIL import Image
from torchvision import transforms

def predict_image(image_path):
    """
    Placeholder function for image model prediction.
    Later this will load K's image_model.pt
    """
    # Example of preprocessing (just placeholder)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    # Open the image (for now just a dummy logic)
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0)
    
    # Dummy output (replace with real model prediction later)
    return "Fake", 0.75
