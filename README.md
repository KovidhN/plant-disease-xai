# Plant Disease XAI

A local plant-leaf decision-support demo combining a trained 38-class
MobileNetV2 classifier, Grad-CAM visualizations, and an optional structured
advisory from local Ollama.

> **Scope:** This is a prototype for demonstration and educational use. It is
> not a validated agricultural diagnosis or a substitute for local expertise.
> PlantVillage performance does not establish accuracy on field images.

## Features

- Upload a JPG/JPEG/PNG leaf image or capture one with a camera.
- Show top-three predictions from a PyTorch MobileNetV2 model.
- Optionally filter and re-normalize probabilities to a selected plant species.
  Confidence is then conditional on that selected species.
- Display the resized image, raw Grad-CAM heatmap, overlay, and activation
  evidence metrics.
- For non-healthy predictions, request a structured advisory from local Ollama.
  The advisory cannot replace the model's predicted class. Healthy predictions
  skip Ollama; failed requests or invalid advisory JSON use deterministic local
  rules.
- Display and optionally play the Hindi summary using gTTS (requires internet).
- Run inference on CPU; CUDA is used automatically when available.

## Requirements

- Windows PowerShell examples below assume the project virtual environment is
  named `venv`.
- Python dependencies are listed in `requirements.txt`.
- A trained checkpoint is required but **is not included in this public
  repository**. Obtain the project checkpoint separately and place it at
  `trained instnace/mobilenetv2_plant_disease.pth`, or set `MODEL_PATH`.
- To enable the live advisory, install and run Ollama and obtain the
  `qwen2.5:3b` model.

## Setup (Windows)

From the project directory:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` only if you need a different Ollama URL/model or checkpoint path.
Never commit `.env` or put API keys in source control. The local Ollama defaults
are:

```text
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
```

Install the model once if needed:

```powershell
ollama pull qwen2.5:3b
```

If Ollama is not already running, start it in a separate terminal:

```powershell
ollama serve
```

The app sends advisory requests to `http://localhost:11434/api/chat` with a
120-second timeout. It reports the request status and uses local fallback rules
if Ollama is unreachable or returns invalid advisory data.

## Run the Streamlit app

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

Open [http://localhost:8501](http://localhost:8501). Keep the terminal open
while using the app; press Ctrl+C in it to stop the server. Check the service
health with:

```powershell
Invoke-RestMethod "http://localhost:8501/_stcore/health"
```

## Other commands

Run the test suite:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

Classify one image from the command line:

```powershell
.\venv\Scripts\python.exe predict.py "C:\path\to\leaf.jpg"
```

The separate legacy FastAPI service is optional and is not required by the
Streamlit dashboard. Its model-loading path is distinct; check that it uses a
real checkpoint rather than mock inference before relying on its predictions.

## Model and preprocessing

The Streamlit model utility loads the 38-class checkpoint with CPU/CUDA device
selection and `map_location`. Images are converted to RGB, resized to 224x224,
and normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard
deviation `[0.229, 0.224, 0.225]`. Class indices must match the order in
`src/class_mapping.py`.

The checkpoint is intentionally excluded from GitHub by `.gitignore`. The
dataset, virtual environment, `.env`, Python caches, and local outputs are also
excluded. Model weights and evaluation data should be managed separately and
only shared when their provenance and permissions are clear.

## Tests and limitations

The repository includes tests for model/API behavior, species filtering,
Grad-CAM, and the decision layer. A passing software test suite does not
establish classifier accuracy. Add verified Colab logs and independent
held-out evaluation metrics to any report; do not infer them from successful
checkpoint loading or demo predictions.

Grad-CAM is a visual aid, not proof of causal reasoning or a measure of disease
severity. LLM-generated advice is preliminary, not independently verified
agronomic guidance or a treatment prescription. Real-world field
generalization has not been established.
