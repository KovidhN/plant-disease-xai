# Plant Disease XAI — Project Handoff

Updated: 2026-11-13

This document is a portable project brief for a new collaborator or coding
assistant. It records the repository state observed on the date above. Treat
unreported training results and files as unknown; do not invent metrics.

## What the project is

Plant Disease XAI is a local demo of a prototype agricultural
decision-support application. The standalone Streamlit flow accepts an image,
classifies it with a 38-class MobileNetV2 model, displays Grad-CAM visual
evidence, and optionally adds a structured advisory through local Ollama.
FastAPI is a separate optional legacy service, not a dependency of the
dashboard.

This is a prototype, not a clinically or agriculturally validated diagnostic
product. PlantVillage images are relatively controlled; performance on
real-world field images is unverified.

## Current status

- The user reports that model training was completed in Google Colab.
- The owner has a local checkpoint at
  `trained instnace/mobilenetv2_plant_disease.pth`. Model weights are
  intentionally excluded from the public Git repository; obtain/place the
  checkpoint separately to run inference.
- The checkpoint loaded successfully on CPU and generated a top-3 prediction
  for a synthetic smoke-test image. This verifies compatibility/inference only,
  not accuracy or real-world performance.
- Colab training logs, validation metrics, independent test results, and exact
  dataset details have not been provided.
- The `models/` directory contains only `.gitkeep`; the standalone app
  discovers the checkpoint in `trained instnace/`, the project root, or
  through the `MODEL_PATH` environment variable.
- The dataset is not present locally under `data/plantvillage/`.
- `app.py` is a standalone Streamlit dashboard supporting file upload or
  camera capture, optional species filtering, top-three predictions,
  raw/overlaid Grad-CAM with XAI metrics, structured advisory, and optional
  Hindi speech output.
- `model_utils.py` loads the checkpoint on CPU/CUDA, normalizes resized RGB
  images, filters/re-normalizes softmax probabilities by species when
  requested, and generates class-specific Grad-CAM overlays and activation
  coverage.
- `decision_layer.py` calls local Ollama, validates advisory JSON, preserves
  the model's predicted class, skips Ollama for healthy predictions, and
  returns a diagnostic status reason with deterministic fallback on failure.
- The local Ollama API and `qwen2.5:3b` were confirmed available. A live
  advisory request succeeded in about 35.5 seconds in one observed run.
- The latest complete pytest run passed 12 tests. This is software-test
  evidence, not classifier-accuracy evidence.
- The checkpoint loaded on CPU and generated a prediction and Grad-CAM
  smoke-test result. This verifies compatibility/execution only.
- Colab training logs, exact training configuration, validation metrics, and
  independent test results were not provided for verification. Field
  generalization is unverified.

## Repository map

- `app.py` — Standalone Streamlit front end. Supports upload/camera, species
  filtering, CPU/CUDA/checkpoint/engine status, top-three predictions,
  Grad-CAM, and expandable structured guidance. It does not require FastAPI.
- `model_utils.py` — Resolves and safely loads the 38-class MobileNetV2
  checkpoint, selects CPU/CUDA with `map_location`, filters probabilities by
  one of 14 species, and generates class-specific Grad-CAM results.
- `decision_layer.py` — Local Ollama advisory, schema validation, fixed-label
  preservation, healthy-result handling, and deterministic fallback.
- `.env.example` — Non-secret local Ollama and optional checkpoint settings.
- `main.py` — FastAPI service. Initializes the model, Grad-CAM explainer, and
  Jev layer at import/startup. Exposes `GET /`, `POST /predict`, and
  `POST /analyze`.
- `src/model.py` — MobileNetV2 architecture, checkpoint loading, prediction,
  and mock fallback.
- `src/preprocessing.py` — RGB conversion, resize/crop to 224x224, ImageNet
  normalization.
- `src/class_mapping.py` — Ordered list of 38 PlantVillage class names. Model
  output indices must use exactly this order.
- `src/gradcam.py` — Grad-CAM generation and overlay; emits a random mock
  heatmap if the model/dependency is unavailable.
- `src/decision.py` — Converts prediction and Grad-CAM coverage into structured
  evidence for Jev.
- `src/jev.py` — External AI Gateway decision layer with local fallback.
- `src/evaluation.py` — Accuracy, weighted precision/recall/F1 and confusion
  matrix helpers.
- `tests/` — Current tests primarily exercise mock inference and mock Grad-CAM;
  they do not establish trained-model accuracy or full real-model integration.
- `README.md`, `SYNOPSIS.md`, `howtooperate.txt` — Setup, operation, and
  project/report context. Check these against current source before citing.

## Next steps: validate the trained model

To complete scientific evaluation, obtain from Colab:

1. The Colab training output: best validation accuracy/loss and epoch, total
   sample counts, dataset source/version, and training command/config.
2. The dataset `class_to_idx` mapping and confirmation that it matches this
   repository's `src/class_mapping.py`.
3. The exact preprocessing used for validation/inference.

The local checkpoint is expected at:

```text
trained instnace/mobilenetv2_plant_disease.pth
```

The standalone app also checks the project root or accepts `MODEL_PATH` from
the process environment. The checkpoint is intentionally excluded from the
public repository. Do not add API keys or other secrets to this handoff,
source control, or shared prompts.

Then continue validation in this order:

1. Obtain the Colab logs and confirm class-index order and preprocessing.
2. Evaluate on a held-out dataset and record accuracy, per-class and aggregate
   precision/recall/F1, latency, and model size. The dataset is not included
   in this repository; evaluate in Colab or configure a local copy.
3. Test the standalone UI with representative leaf images.
4. If using the optional API, configure its separate checkpoint path and
   verify `GET /`, `POST /predict`, and `POST /analyze`; the API may otherwise
   fall back to mock inference.
5. Record results and limitations in the README or a dedicated evaluation
   report. Do not claim generalization to field images based only on
   PlantVillage validation.

Preserve the same class-index order for evaluation as the trained checkpoint.

## Known issues and validation cautions

1. **Species-filter confidence is conditional.** When selected, probabilities
   are re-normalized over that species only; users should know the filter is a
   prior assumption and can elevate the displayed confidence.
2. **Legacy API can still return mock output.** `main.py` defaults to
   `models/mobilenetv2_plant_disease.pth`, whereas the provided checkpoint is in
   `trained instnace/`. The standalone UI uses the actual checkpoint; API
   callers must configure/copy it or treat `is_mock` as non-diagnostic.
3. **Mock paths look superficially successful.** API mock predictions are random,
   and mock Grad-CAM is a random heatmap. Surface `is_mock` clearly in the
   UI/API and never use mock results as evidence.
4. **Invalid image handling needs hardening.** `preprocess_image()` catches
   errors and returns `None`, while the API assumes a valid result. Add a
   clear client error (HTTP 400/415) for invalid or unsupported images.
5. **Legacy Jev live integration needs verification.** Inspect `src/jev.py` before
   enabling it: the current checked-in authorization header does not visibly
   interpolate the configured API key. Verify the header, endpoint, model, and
   response schema using environment variables; never hard-code credentials.
   The Jev layer is advisory and must not invent treatment instructions.
6. **Checkpoint loading silently falls back.** `src/model.py` logs a load
   failure and continues with mock inference. For deployment, expose that state
   clearly and consider failing startup when a required model is configured
   but cannot load.
7. **Tests are insufficient for a trained release.** Add a real-checkpoint
   load/inference test (skipped when no test checkpoint is available), API
   invalid-file tests, and an `/analyze` integration test. Avoid checking in
   large weights or private datasets without an explicit decision.

## One-day completion priorities

1. Verify training logs, class mapping, and preprocessing against the local
   checkpoint.
2. Evaluate on a held-out set and capture accuracy, weighted precision/recall,
   F1, latency, and model size; clearly label the split and limitations.
3. Test real-image inference and Grad-CAM in the API if that service is part
   of the deliverable.
4. Fix any checkpoint/label mismatch and handle
   invalid uploads.
5. Run tests, launch API/UI from a clean environment, and rehearse the demo.
6. Defer PlantDoc/PlantWild fine-tuning, mobile deployment, temporal analysis,
   and other future-scope work unless the core validation finishes early.

## Commands

Run from the project root with the documented dependencies installed:

```bash
streamlit run app.py
uvicorn main:app --reload
pytest
```

The optional API has separate checkpoint and mock-inference behavior; verify
its model status before relying on its predictions. Do not rerun training
unless there is a specific reason; the next milestone is collecting evaluation
evidence and validating representative images.

## Local advisory setup

Copy `.env.example` to `.env`. The standalone dashboard calls local Ollama at
`http://localhost:11434/api/chat` and defaults to `qwen2.5:3b`. Healthy
predictions skip Ollama; non-healthy predictions use a 120-second request
timeout and fall back to deterministic local rules on request or response
failure. The dashboard's `decision_layer.py` is distinct from the legacy
`src/jev.py` integration used by the optional API.

## Instructions for the next AI assistant

Continue from the current repository state; inspect files before editing.
The checkpoint is already in the workspace and compatible with the standalone
app. First request the Colab logs and evaluation evidence. Do not fabricate
metrics, treat legacy API mock output as real, or expose credentials. Keep
changes surgical, run focused tests, and update documentation when behavior
changes. Favor a credible, reproducible prototype over unsupported claims of
field readiness.
