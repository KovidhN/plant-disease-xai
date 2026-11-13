import unittest
from src.model import PlantDiseaseModel
from src.class_mapping import PLANTVILLAGE_CLASSES

class TestPlantDiseaseModel(unittest.TestCase):
    def setUp(self):
        self.model = PlantDiseaseModel()
        
    def test_mock_inference(self):
        result = self.model.predict(None)
        self.assertTrue(result["is_mock"])
        self.assertIn(result["predicted_class"], PLANTVILLAGE_CLASSES)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertEqual(len(result["top_k"]), 3)

if __name__ == "__main__":
    unittest.main()
