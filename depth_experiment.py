import cv2
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

IMG_ROOT = Path(
    "/agroscope/Data-Work-CH/22_Plant_Production-CH/"
    "224_Digitalisation/Jonas_Anderegg_Files/B_Data/"
    "04_DL_datasets_updates/symptoms/"
)

OUT_ROOT = Path("/agroscope/Data-Work-CH/22_Plant_Production-CH/"
    "224_Digitalisation/Jonas_Anderegg_Files/E_Work/91_DL/focus_segmentation/output/scale100")

DOWNSCALE_FACTOR = 1
CROPPING = False


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}

encoder = "vits"

model = DepthAnythingV2(**model_configs[encoder])

model.load_state_dict(
    torch.load(
        f"./models/depth_anything_v2_{encoder}.pth",
        map_location="cpu",
    )
)

model = model.to(DEVICE).eval()


# ---------------------------------------------------------------------
# Find batch directories
# ---------------------------------------------------------------------

batch_dirs = sorted(
    p for p in IMG_ROOT.iterdir()
    if p.is_dir() and p.name.startswith("batch")
)

print(f"Found {len(batch_dirs)} batch directories")


# ---------------------------------------------------------------------
# Process images
# ---------------------------------------------------------------------

for batch_dir in batch_dirs[:1]:

    img_dir = batch_dir / "img"

    if not img_dir.is_dir():
        print(f"Skipping {batch_dir.name}: no img directory")
        continue

    image_paths = sorted(img_dir.glob("*.JPG"))

    print(
        f"\n{batch_dir.name}: "
        f"found {len(image_paths)} JPG images"
    )

    # Output:
    # ./output/batchX/img/
    out_img_dir = OUT_ROOT / batch_dir.name / "img"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths, start=1):

        print(
            f"[{i}/{len(image_paths)}] "
            f"{batch_dir.name}/{img_path.name}"
        )

        # -------------------------------------------------------------
        # Read image
        # -------------------------------------------------------------

        raw_img = cv2.imread(str(img_path))

        if raw_img is None:
            print(f"  WARNING: could not read {img_path}")
            continue

        # -------------------------------------------------------------
        # Depth inference
        # -------------------------------------------------------------
        # gather sizes/offsets from instance or globals
        csx1, csx2 = [6144, 4096]
        cox1, cox2 = [1750, 500]

        img_rotate = raw_img
        if raw_img.shape == (5464, 8192, 3):
            img_rotate = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)

        # perform cropping
        if CROPPING: 
            y1 = int(cox1)
            x1 = int(cox2)
            y2 = y1 + int(csx1)
            x2 = x1 + int(csx2)
            if 0 <= y1 < y2 <= img_rotate.shape[0] and 0 <= x1 < x2 <= img_rotate.shape[1]:
                img_crop = img_rotate[y1:y2, x1:x2, :]
        else:
            img_crop = img_rotate

        h, w = img_crop.shape[:2]
        resized_img = cv2.resize(
            img_crop,
            (w // DOWNSCALE_FACTOR, h // DOWNSCALE_FACTOR),
            interpolation=cv2.INTER_AREA,
            )

        with torch.no_grad():
            depth = model.infer_image(img_crop, )

        # -------------------------------------------------------------
        # Save raw relative depth
        # -------------------------------------------------------------

        depth_path = out_img_dir / f"{img_path.stem}.npy"

        np.save(depth_path, depth)

        # -------------------------------------------------------------
        # Normalize depth for visualization
        # -------------------------------------------------------------

        min_val = depth.min()
        max_val = depth.max()

        if max_val > min_val:
            normalized_depth = (
                (depth - min_val)
                / (max_val - min_val)
            )
        else:
            normalized_depth = np.zeros_like(depth)

        # -------------------------------------------------------------
        # Save visualization
        # -------------------------------------------------------------

        vis_path = out_img_dir / f"{img_path.stem}_depth.png"

        plt.figure(figsize=(6, 9), dpi=320)
        plt.imshow(normalized_depth, cmap="Spectral_r")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            vis_path,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()


print("\nFinished.")

