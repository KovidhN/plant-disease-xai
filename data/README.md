# Dataset Strategy

## Primary Dataset
**A. PlantVillage** ⭐⭐⭐⭐⭐
- **Role:** Primary dataset for training, validation, benchmarking, initial MobileNetV2 modeling, and Grad-CAM evaluation.
- **Details:** Contains 38 classes of plant leaf diseases. 

## Secondary & Generalization Datasets
To test how well the model generalizes outside of controlled laboratory conditions, the following datasets are shortlisted for future experiments:

**B. PlantDoc** ⭐⭐⭐⭐⭐
- **Role:** Real-world generalization testing.
- **Details:** Images are much closer to those encountered outside a controlled environment.

**C. PlantWild** ⭐⭐⭐⭐
- **Role:** In-the-wild plant disease recognition.
- **Details:** Useful for testing performance against varied backgrounds, lighting, and environments.

**D. Rice Leaf Disease Datasets** ⭐⭐⭐⭐
- **Role:** Crop-specific secondary experiment.
- **Details:** Demonstrates the pipeline isn't conceptually limited to just PlantVillage crops.

**E. Cassava Leaf Disease** ⭐⭐⭐⭐
- **Role:** Real-world agricultural use case.
- **Details:** Good for demonstrating the full classification → explainability → decision support pipeline on a different crop.

**F. Citrus Leaves** ⭐⭐⭐
- **Role:** Transfer-learning/generalization experiments.
