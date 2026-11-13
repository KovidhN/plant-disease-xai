import numpy as np
import cv2
from PIL import Image

try:
    import torch
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

class GradCAMExplainer:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self.cam = None

        # Try to initialize GradCAM if torch is available and model is loaded
        if GRADCAM_AVAILABLE and self.model_wrapper.model_loaded and self.model_wrapper.model is not None:
            # For MobileNetV2, the final convolutional feature layer is features[-1]
            target_layers = [self.model_wrapper.model.features[-1]]
            # We use use_cuda=False because our deployment constraint is CPU inference
            self.cam = GradCAM(model=self.model_wrapper.model, target_layers=target_layers)

    def generate_heatmap(self, image_tensor, target_class=None):
        """
        Generates the Grad-CAM heatmap for a given image tensor.
        Returns a (224, 224) numpy array containing values between 0 and 1.
        """
        if not GRADCAM_AVAILABLE or not self.model_wrapper.model_loaded or self.cam is None:
            # Mock heatmap if model isn't loaded or dependencies are missing
            return np.random.rand(224, 224).astype(np.float32)

        # targets can be None, in which case it uses the highest scoring category
        targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None
        
        # Generate CAM
        # image_tensor shape must be (1, 3, 224, 224)
        grayscale_cam = self.cam(input_tensor=image_tensor, targets=targets)
        
        # Grayscale cam is returned as (batch_size, width, height)
        heatmap = grayscale_cam[0, :]
        return heatmap

    def overlay_heatmap(self, original_image_pil, heatmap):
        """
        Overlays the generated heatmap onto the original PIL image.
        Returns the blended image as a PIL Image.
        """
        if not GRADCAM_AVAILABLE:
            return original_image_pil

        # Convert PIL image to numpy array in RGB format (normalized 0-1)
        rgb_img = np.array(original_image_pil, dtype=np.float32) / 255.0
        
        # Resize rgb_img to match heatmap if necessary
        if rgb_img.shape[:2] != heatmap.shape:
            rgb_img = cv2.resize(rgb_img, (heatmap.shape[1], heatmap.shape[0]))
        
        # Apply colormap using grad-cam's utility function
        visualization = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        
        # Convert back to PIL Image
        return Image.fromarray(visualization)
