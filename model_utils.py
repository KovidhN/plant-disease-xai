"""Load the trained PlantVillage MobileNetV2 model and run predictions."""

import logging
import os
import threading
from pathlib import Path

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.class_mapping import PLANTVILLAGE_CLASSES

logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATHS = (
    PROJECT_DIR / "mobilenetv2_plant_disease.pth",
    PROJECT_DIR / "trained instnace" / "mobilenetv2_plant_disease.pth",
)

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
PLANT_SPECIES = tuple(sorted({name.split("___", 1)[0] for name in PLANTVILLAGE_CLASSES}))


def resolve_checkpoint_path(checkpoint_path: str | Path | None = None) -> Path:
    """Resolve an explicit path, MODEL_PATH, or a known project checkpoint path."""
    environment_path = os.getenv("MODEL_PATH") if checkpoint_path is None else None
    configured_path = checkpoint_path or environment_path
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_DIR / path
        if path.is_file():
            return path
        if checkpoint_path is not None:
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        logger.warning(
            "MODEL_PATH checkpoint not found: %s; searching known checkpoint locations",
            path,
        )

    for path in DEFAULT_CHECKPOINT_PATHS:
        if path.is_file():
            return path

    searched_paths = "\n".join(f"- {path}" for path in DEFAULT_CHECKPOINT_PATHS)
    raise FileNotFoundError(
        "Could not find the trained model checkpoint. Checked:\n"
        f"{searched_paths}\n"
        "Set MODEL_PATH to the checkpoint's location if it is elsewhere."
    )


class PlantDiseasePredictor:
    """CPU/CUDA MobileNetV2 predictor for the 38 PlantVillage classes."""

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        path = resolve_checkpoint_path(checkpoint_path)

        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(
            model.last_channel, len(PLANTVILLAGE_CLASSES)
        )

        # map_location makes CUDA-trained checkpoints load on CPU-only machines.
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            checkpoint = checkpoint.get(
                "state_dict", checkpoint.get("model_state_dict", checkpoint)
            )
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Unsupported checkpoint format: {path}")

        if checkpoint and all(key.startswith("module.") for key in checkpoint):
            checkpoint = {
                key.removeprefix("module."): value for key, value in checkpoint.items()
            }

        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as error:
            raise RuntimeError(
                f"Checkpoint is incompatible with the 38-class MobileNetV2 model: {path}"
            ) from error

        self.model = model.to(self.device)
        self.model.eval()
        self._inference_lock = threading.RLock()
        self._grad_cam = GradCAM(
            model=self.model,
            target_layers=[self.model.features[-1]],
        )

    def _prepare_image(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")

        resized_image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        input_tensor = IMAGE_TRANSFORM(resized_image).unsqueeze(0).to(self.device)
        return resized_image, input_tensor

    def predict(
        self,
        image: Image.Image,
        plant_filter: str | None = None,
    ) -> list[dict[str, str | float]]:
        """Return the top three classes, optionally renormalized within a species."""
        _, input_tensor = self._prepare_image(image)
        with self._inference_lock, torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor)[0], dim=0)
        return self._rank_predictions(probabilities, plant_filter)

    def _generate_gradcam(
        self,
        resized_image: Image.Image,
        input_tensor: torch.Tensor,
        target_class_index: int,
        target_class_label: str,
    ) -> tuple[Image.Image, dict[str, object]]:
        """Return the blended overlay and metrics computed from one normalized CAM.

        The method and target-layer strings identify the explainer; target class
        records the exact backpropagation index and label. Peak and mean activation
        summarize normalized CAM intensity; coverage is the percent of pixels
        above 0.5. Center focus is the percent of those pixels in the centered 70%
        crop. The standalone raw heatmap is returned separately from the overlay.
        """
        rgb_image = np.asarray(resized_image, dtype=np.float32) / 255.0
        with self._inference_lock:
            raw_cam = self._grad_cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(target_class_index)],
            )[0]
        cam = np.asarray(raw_cam, dtype=np.float32)
        if cam.shape != rgb_image.shape[:2]:
            cam = cv2.resize(cam, (rgb_image.shape[1], rgb_image.shape[0]))
        cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam_min = np.float32(cam.min())
        cam_max = np.float32(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        cam = np.clip(cam, np.float32(0.0), np.float32(1.0)).astype(
            np.float32, copy=False
        )

        high_activation = cam > np.float32(0.5)
        active_pixel_count = int(np.count_nonzero(high_activation))
        height, width = cam.shape
        crop_height = int(round(height * 0.7))
        crop_width = int(round(width * 0.7))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        center_activation = high_activation[
            top : top + crop_height, left : left + crop_width
        ]
        center_pixel_count = int(np.count_nonzero(center_activation))

        colorized_bgr = cv2.applyColorMap(
            (cam * np.float32(255.0)).astype(np.uint8),
            cv2.COLORMAP_JET,
        )
        raw_heatmap = Image.fromarray(cv2.cvtColor(colorized_bgr, cv2.COLOR_BGR2RGB))
        overlay = show_cam_on_image(rgb_image, cam, use_rgb=True)
        xai_metrics: dict[str, object] = {
            "method": "Grad-CAM",
            "target_layer": "MobileNetV2.features[-1]",
            "target_class": {
                "index": int(target_class_index),
                "label": str(target_class_label),
            },
            "peak_activation": float(cam.max()),
            "mean_activation": float(cam.mean()),
            "coverage_pct": float(active_pixel_count / cam.size * 100.0),
            "center_focus_pct": float(
                center_pixel_count / active_pixel_count * 100.0
                if active_pixel_count
                else 0.0
            ),
            "raw_heatmap_img": raw_heatmap,
        }
        return Image.fromarray(overlay), xai_metrics

    @staticmethod
    def _rank_predictions(
        probabilities: torch.Tensor,
        plant_filter: str | None = None,
    ) -> list[dict[str, str | float]]:
        if plant_filter is not None and plant_filter not in PLANT_SPECIES:
            raise ValueError(f"Unknown plant species filter: {plant_filter}")

        if plant_filter:
            allowed_indices = [
                index
                for index, name in enumerate(PLANTVILLAGE_CLASSES)
                if name.split("___", 1)[0] == plant_filter
            ]
            selected = probabilities[allowed_indices]
            selected = selected / selected.sum()
            top_probabilities, local_indices = torch.topk(
                selected, k=min(3, len(allowed_indices))
            )
            top_indices = torch.tensor(
                [allowed_indices[index] for index in local_indices.tolist()],
                device=probabilities.device,
            )
        else:
            top_probabilities, top_indices = torch.topk(
                probabilities, k=min(3, len(PLANTVILLAGE_CLASSES))
            )

        return [
            {
                "class_name": PLANTVILLAGE_CLASSES[index],
                "confidence": float(probability) * 100,
            }
            for probability, index in zip(
                top_probabilities.cpu().tolist(), top_indices.cpu().tolist()
            )
        ]

    def explain(
        self,
        image: Image.Image,
        target_class_index: int,
    ) -> tuple[Image.Image, dict[str, object]]:
        """Return a class-targeted overlay and its Grad-CAM evidence metrics."""
        if not 0 <= target_class_index < len(PLANTVILLAGE_CLASSES):
            raise ValueError(f"Unknown target class index: {target_class_index}")
        resized_image, input_tensor = self._prepare_image(image)
        return self._generate_gradcam(
            resized_image,
            input_tensor,
            target_class_index,
            PLANTVILLAGE_CLASSES[target_class_index],
        )

    def analyze(
        self,
        image: Image.Image,
        plant_filter: str | None = None,
    ) -> dict[str, object]:
        """Return ranked predictions and a Grad-CAM for the top filtered class."""
        resized_image, input_tensor = self._prepare_image(image)
        with self._inference_lock:
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(input_tensor)[0], dim=0)
            predictions = self._rank_predictions(probabilities, plant_filter)
            predicted_class = str(predictions[0]["class_name"])
            target_index = PLANTVILLAGE_CLASSES.index(predicted_class)
        overlay, xai_metrics = self._generate_gradcam(
            resized_image,
            input_tensor,
            target_index,
            predicted_class,
        )
        return {
            "predictions": predictions,
            "gradcam_overlay": overlay,
            "gradcam_coverage": xai_metrics["coverage_pct"],
            "xai_metrics": xai_metrics,
            "resized_image": resized_image,
        }
