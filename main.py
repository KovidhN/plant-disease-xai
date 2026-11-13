import os
import time
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image
from src.model import PlantDiseaseModel
from src.preprocessing import preprocess_image
from src.gradcam import GradCAMExplainer
from src.jev import JevDecisionLayer
from src.decision import formulate_structured_evidence

load_dotenv()

app = FastAPI(title="Plant Disease Detection API", version="0.1.0")

# Initialize components
MODEL_PATH = os.getenv("MODEL_PATH", "models/mobilenetv2_plant_disease.pth")
AI_GATEWAY_API_KEY = os.getenv("AI_GATEWAY_API_KEY")

model = PlantDiseaseModel(model_path=MODEL_PATH)
explainer = GradCAMExplainer(model_wrapper=model)
jev = JevDecisionLayer(api_key=AI_GATEWAY_API_KEY)

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_k: list
    is_mock: bool
    latency_ms: float

class AnalysisResponse(BaseModel):
    prediction: dict
    jev_decision: dict
    heatmap_base64: str
    latency_ms: float

@app.get("/")
def read_root():
    return {"status": "running", "model_loaded": model.model_loaded}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    
    contents = await file.read()
    preprocessed = preprocess_image(contents)
    
    prediction = model.predict(preprocessed["tensor"])
    
    latency_ms = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        top_k=prediction["top_k"],
        is_mock=prediction["is_mock"],
        latency_ms=latency_ms
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    start_time = time.time()
    
    contents = await file.read()
    
    # 1. Preprocess
    preprocessed = preprocess_image(contents)
    
    # 2. Predict
    prediction = model.predict(preprocessed["tensor"])
    
    # 3. Grad-CAM
    heatmap = explainer.generate_heatmap(preprocessed["tensor"])
    
    # Calculate basic coverage stats
    coverage = float(heatmap.mean()) if hasattr(heatmap, 'mean') else 0.45
    gradcam_stats = {"coverage": coverage}
    
    # Overlay and encode base64
    overlay_img = explainer.overlay_heatmap(preprocessed["image"], heatmap)
    
    if isinstance(overlay_img, Image.Image):
        buffered = BytesIO()
        overlay_img.save(buffered, format="JPEG")
        heatmap_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:
        # Fallback if mock returns something else
        heatmap_base64 = ""
    
    # 4. Formulate structured evidence
    evidence = formulate_structured_evidence(prediction, gradcam_stats)
    
    # 5. Jev Decision
    jev_decision = jev.evaluate(evidence)
    
    latency_ms = (time.time() - start_time) * 1000
    
    return AnalysisResponse(
        prediction=prediction,
        jev_decision=jev_decision,
        heatmap_base64=heatmap_base64,
        latency_ms=latency_ms
    )
