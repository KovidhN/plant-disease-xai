# Plant Disease XAI — Project Synopsis

## Project summary

Plant Disease XAI is a locally runnable student-project demo for plant-leaf
image classification and decision support. Its standalone Streamlit dashboard
uses a supplied PyTorch MobileNetV2 checkpoint for 38 PlantVillage classes,
Grad-CAM visual evidence, and an optional structured advisory from local
Ollama. It is a prototype, not a validated agricultural diagnostic product.

## Objectives

- Provide a simple local interface for uploading or capturing a leaf image.
- Display the model's top-three class predictions.
- Provide visual context for a prediction using Grad-CAM.
- Present cautious structured guidance without allowing the advisory to change
  the classifier's predicted label.
- Support CPU inference without requiring a GPU.

## System flow

1. The user uploads JPG/JPEG/PNG or captures a leaf image in the Streamlit app.
2. The image is converted to RGB, resized to 224x224, and normalized using
   ImageNet mean and standard deviation.
3. MobileNetV2 generates class probabilities. An optional species filter
   re-normalizes scores among that species' classes; confidence is therefore
   conditional on the filter.
4. Class-targeted Grad-CAM on `MobileNetV2.features[-1]` produces a raw
   heatmap, overlay, and activation metrics.
5. For non-healthy predictions, structured model evidence is sent to local
   Ollama (default model `qwen2.5:3b`) for an advisory. Healthy predictions
   skip Ollama. On request or response failure, deterministic local rules are
   used.
6. The Hindi summary can optionally be read as audio using gTTS, which
   requires network access.

The Streamlit app is standalone. The separate legacy FastAPI service is
optional and has a distinct model-loading path; its mock status should be
checked before using its predictions.

## Current demonstration status

The local dashboard has been smoke-tested with the trained checkpoint on CPU.
A local Ollama endpoint and `qwen2.5:3b` were confirmed available, and one live
advisory request completed successfully in approximately 35.5 seconds.
The latest complete software test run passed 12 tests. These checks show that
the application can execute its demonstrated flow; they do not establish
classifier accuracy or field performance.

## Limitations

- The Colab training logs, exact dataset version/counts, training split,
  configuration, validation metrics, and independent test results have not
  been provided for verification.
- PlantVillage imagery is controlled; generalization to real field conditions
  is unverified.
- Species-filter confidence is conditional on the user-selected species.
- Grad-CAM is a visual aid, not causal proof or a disease-severity metric.
- Ollama advice is generated text, not independently verified agricultural
  guidance or a treatment prescription.
- gTTS speech generation requires internet access.
- The model checkpoint and dataset are not included in the public repository;
  obtain authorized copies separately.

## Future work

- Recover training logs and verify the class mapping and preprocessing used.
- Evaluate on a documented independent test set and report per-class and
  aggregate metrics, confusion matrix, and inference latency.
- Assess performance on representative field images and document failure
  cases before making deployment or generalization claims.
- Evaluate the optional FastAPI service independently, including whether it
  loaded the trained checkpoint or returned mock results.
