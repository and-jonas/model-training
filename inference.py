
from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf

from transforms_pytorch import build_val_transform


MODEL_PATH = (
    Path.home()
    / "leaf-toolkit"
    / "models"
    / "seg_fpn_mitb3_tkbkocy7.torchscript"
)

IMG_PATH = Path(
    "/agroscope/Data-Work-CH/22_Plant_Production-CH/"
    "224_Digitalisation/Jonas_Anderegg_Files/B_Data/"
    "01_DL_Datasets/03_CANOPY_PROXIMAL_SYMPTOMS/"
    "iter2/dataset_src/segmentations_export/data"
)

OUTPUT_DIR = Path("./evaluation_torchscript/predictions")

AUGMENTATION_CONFIG_PATH = (
    "config/symptoms/data_augmentation/base.yaml"
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


# Load model
model = torch.jit.load(
    str(MODEL_PATH),
    map_location=DEVICE,
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