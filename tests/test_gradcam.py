import unittest
import numpy as np
from src.gradcam import GradCAMExplainer
from src.model import PlantDiseaseModel

class TestGradCAM(unittest.TestCase):
    def setUp(self):
        self.model = PlantDiseaseModel()
        self.explainer = GradCAMExplainer(self.model)
        
    def test_generate_heatmap(self):
        heatmap = self.explainer.generate_heatmap(None)
        self.assertEqual(heatmap.shape, (224, 224))

if __name__ == "__main__":
    unittest.main()
