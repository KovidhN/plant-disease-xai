"""Run the trained PlantVillage classifier on an image from the command line."""

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from model_utils import PlantDiseasePredictor


def readable_class_name(class_name: str) -> str:
    """Format a PlantVillage class label for terminal output."""
    return " ".join(class_name.replace("___", " - ").replace("_", " ").split())


def predict_leaf(
    image_path: str | Path,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, str | float]]:
    """Print and return the top-three predictions for one image."""
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image file '{path}': {error}") from error

    predictor = PlantDiseasePredictor(checkpoint_path=checkpoint_path)
    predictions = predictor.predict(image)

    print(f"Device: {predictor.device.type.upper()}")
    print(f"Image: {path.resolve()}")
    print("Top predictions:")
    for rank, prediction in enumerate(predictions, start=1):
        label = readable_class_name(str(prediction["class_name"]))
        confidence = float(prediction["confidence"])
        print(f"{rank}. {label} — {confidence:.2f}%")

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a plant leaf image with the trained MobileNetV2 model."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a JPG, JPEG, or PNG leaf image",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional model checkpoint path. Defaults to MODEL_PATH or the "
            "project's known checkpoint locations."
        ),
    )
    args = parser.parse_args()

    try:
        predict_leaf(args.image, checkpoint_path=args.checkpoint)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
