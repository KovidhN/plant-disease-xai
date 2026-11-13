import torch

from model_utils import PLANT_SPECIES, PlantDiseasePredictor
from src.class_mapping import PLANTVILLAGE_CLASSES


def test_species_list_matches_plantvillage_classes():
    assert len(PLANT_SPECIES) == 14
    assert all(species in PLANT_SPECIES for species in ("Apple", "Corn_(maize)", "Tomato"))


def test_species_filter_renormalizes_only_its_classes():
    probabilities = torch.arange(1, len(PLANTVILLAGE_CLASSES) + 1, dtype=torch.float32)
    predictions = PlantDiseasePredictor._rank_predictions(probabilities, "Apple")

    assert len(predictions) == 3
    assert all(item["class_name"].startswith("Apple___") for item in predictions)
    assert abs(sum(item["confidence"] for item in predictions) - 90.0) < 1e-4
    assert all(0 <= item["confidence"] <= 100 for item in predictions)
