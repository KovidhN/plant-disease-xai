from PIL import Image
import io
import torch
from torchvision import transforms

def preprocess_image(image_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # MobileNetV2 ImageNet standard transforms
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Add batch dimension (B, C, H, W)
        tensor = preprocess(img).unsqueeze(0)
        
        return {"tensor": tensor, "original_size": img.size, "image": img}
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
