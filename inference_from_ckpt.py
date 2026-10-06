from lightning_model import SegmentationModel

from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf

from transforms_pytorch import build_val_transform

MODEL_PATH = ("loggers/focus_mitb3_fpn/version_0/checkpoints/epochepoch=22-valIoUIoU=0.8656.ckpt")

IMG_PATH = Path(
    "/agroscope/Data-Work-CH/22_Plant_Production-CH/"
    "224_Digitalisation/Jonas_Anderegg_Files/B_Data/"
    "01_DL_Datasets/05_CANOPY_PROXIMAL_FOCUS/"
    "dataset_src/segmentations_export/data"
)

OUTPUT_DIR = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/"
    "224_Digitalisation/Jonas_Anderegg_Files/E_Work/91_DL/05_CANOPY_PROXIMAL_FOCUS/predictions")

AUGMENTATION_CONFIG_PATH = (
    "config/focus/data_augmentation/base.yaml"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load validation transform
augmentation_cfg = OmegaConf.load(
    AUGMENTATION_CONFIG_PATH
)

transform = build_val_transform(
    augmentation_cfg
)


# Load model from Lightning checkpoint
model = SegmentationModel.load_from_checkpoint(
    str(MODEL_PATH),
    map_location=DEVICE,
    weights_only=False,
)

model.eval()
model.to(DEVICE)

# Create output directory
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Predict
image_paths = sorted(
    IMG_PATH.glob("*.png")
)

with torch.no_grad():

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        transformed = transform(
            image=image,
        )

        image = transformed["image"]

        image_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        outputs = model(image_tensor)

        predictions = torch.argmax(
            outputs,
            dim=1,
        )

        prediction_mask = (
            predictions[0]
            .cpu()
            .numpy()
            .astype("uint8")
        )

        cv2.imwrite(
            str(OUTPUT_DIR / image_path.name),
            prediction_mask,
        )

        print(image_path.name)