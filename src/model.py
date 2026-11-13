import os
import torch
import torch.nn as nn
from torchvision import models
import random
from src.class_mapping import PLANTVILLAGE_CLASSES

class PlantDiseaseModel:
    def __init__(self, model_path=None):
        # Deployment requirement: CPU inference
        self.device = torch.device("cpu")
        self.model_loaded = False
        self.num_classes = len(PLANTVILLAGE_CLASSES)
        
        # Build the model architecture
        self.model = self._build_model()
        
        # Load weights if available, else fallback to mock mode
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("WARNING: DEMO MODE — trained model weights not loaded.")
            self.model_loaded = False

    def _build_model(self):
        # Load pretrained MobileNetV2
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Replace the classifier layer with our class count
        model.classifier[1] = nn.Linear(model.last_channel, self.num_classes)
        model.to(self.device)
        model.eval()  # Set to evaluation mode immediately
        return model

    def _load_model(self, path):
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model_loaded = True
            print(f"Model loaded successfully from {path}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            self.model_loaded = False
        
    def predict(self, image_tensor=None):
        if not self.model_loaded or image_tensor is None:
            # Fallback to mock inference when weights aren't loaded
            return self._mock_predict()
            
        # Real Inference
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Extract top-k
            top_prob, top_indices = torch.topk(probabilities, 3)
            
            top_k = []
            for i in range(3):
                top_k.append({
                    "class_name": PLANTVILLAGE_CLASSES[top_indices[i].item()],
                    "confidence": round(float(top_prob[i].item()), 4)
                })
            
            return {
                "predicted_class": PLANTVILLAGE_CLASSES[top_indices[0].item()],
                "confidence": round(float(top_prob[0].item()), 4),
                "top_k": top_k,
                "is_mock": False
            }
            
    def _mock_predict(self):
        # Mock logic (same as before)
        class_idx = random.randint(0, self.num_classes - 1)
        confidence = random.uniform(0.6, 0.99)
        top_k = []
        for i in range(3):
            top_k.append({
                "class_name": PLANTVILLAGE_CLASSES[(class_idx + i) % self.num_classes],
                "confidence": round(confidence if i == 0 else random.uniform(0.1, 0.5), 4)
            })
        
        return {
            "predicted_class": PLANTVILLAGE_CLASSES[class_idx],
            "confidence": round(confidence, 4),
            "top_k": top_k,
            "is_mock": True
        }
