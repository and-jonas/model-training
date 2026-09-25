import sys
import torch
import cv2
import albumentations
import segmentation_models_pytorch as smp
import torchmetrics
import wandb
import fiftyone as fo
import hydra
import omegaconf

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("OpenCV:", cv2.__version__)
print("Albumentations:", albumentations.__version__)
print("SMP:", smp.__version__)
print("TorchMetrics:", torchmetrics.__version__)
print("FiftyOne:", fo.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

### TEST TRANSFORMS

import cv2
import numpy as np
from pathlib import Path

from transforms_pytorch import build_train_transform

image_path = next(
    Path("/scratch0/f89601470/ZenklEtAl2026/train/data").glob("*.png")
)

mask_path = Path(
    str(image_path).replace("/data/", "/labels/")
)

image = cv2.imread(str(image_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

print("Before:")
print("image:", image.shape, image.dtype, image.min(), image.max())
print("mask:", mask.shape, mask.dtype, np.unique(mask))

# For this test, we'll load the config directly
from omegaconf import OmegaConf

aug_cfg = OmegaConf.load(
    "config/symptoms/data_augmentation/base.yaml"
)

transform = build_train_transform(aug_cfg)

result = transform(image=image, mask=mask)

image_t = result["image"]
mask_t = result["mask"]

print("\nAfter Albumentations:")
print("image:", image_t.shape, image_t.dtype, image_t.min(), image_t.max())
print("mask:", mask_t.shape, mask_t.dtype, np.unique(mask_t))


### Test dataset

from dataset import SegmentationDataset
from transforms_pytorch import build_train_transform
from omegaconf import OmegaConf

aug_cfg = OmegaConf.load(
    "config/symptoms/data_augmentation/base.yaml"
)

transform = build_train_transform(aug_cfg)

dataset = SegmentationDataset(
    data_dir="/scratch0/f89601470/ZenklEtAl2026/train/data",
    labels_dir="/scratch0/f89601470/ZenklEtAl2026/train/labels",
    transform=transform,
)

print("Dataset size:", len(dataset))

image, mask = dataset[0]

print("Image:", image.shape, image.dtype, image.min(), image.max())
print("Mask:", mask.shape, mask.dtype, mask.unique())


## DataLoader

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

images, masks = next(iter(train_loader))

print("Images:", images.shape, images.dtype)
print("Masks:", masks.shape, masks.dtype)
print("Mask classes:", masks.unique())


### MODEL + LOSS on a real batch
import torch
import segmentation_models_pytorch as smp
from torch.nn import CrossEntropyLoss

device = "cuda:0"

model = smp.FPN(
    encoder_name="mit_b3",
    encoder_weights=None,
    in_channels=3,
    classes=5,
).to(device)

images = images.to(device)
masks = masks.to(device)

print("Running forward pass...")

with torch.no_grad():
    outputs = model(images)
    print("Outputs:", outputs.shape)

    class_weights = torch.tensor(
        [1, 1, 1, 1, 1],
        dtype=torch.float32,
        device=device,
    )
    class_weights = class_weights / class_weights.sum()

    loss_fn = CrossEntropyLoss(weight=class_weights)
    loss = loss_fn(outputs, masks)

print("Loss:", loss.item())



## Training STep

import torch

model.train()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0,
)

optimizer.zero_grad()

outputs = model(images)
loss = loss_fn(outputs, masks)

print("Loss before backward:", loss.item())

loss.backward()

optimizer.step()

print("Training step completed")



## Gradient accumulation

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0,
)

model.train()
optimizer.zero_grad()

accumulate_steps = 2

for step, (batch_images, batch_masks) in enumerate(train_loader):
    batch_images = batch_images.to(device)
    batch_masks = batch_masks.to(device)

    outputs = model(batch_images)
    loss = loss_fn(outputs, batch_masks)

    # Scale loss because gradients are accumulated over 2 minibatches
    loss = loss / accumulate_steps
    loss.backward()

    print(f"Minibatch {step + 1}, scaled loss: {loss.item():.4f}")

    if (step + 1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        print("Optimizer step")
        break

## Train

import os
print(os.environ.get("TORCH_HOME"))
path = "/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth"

print(os.path.exists(path))
print(os.path.getsize(path) / 1024**2)


import torch
import time

t = time.time()

state = torch.load(
    "/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth",
    map_location="cpu",
    weights_only=False,
)

print("loaded in", time.time() - t, "seconds")
print(type(state))

import segmentation_models_pytorch as smp

# initialize model
model = smp.FPN(
    encoder_name="mit_b3",
    encoder_weights=None,  # Wichtig: Auf None setzen!
    in_channels=3,
    classes=5,
)
local_weights_path = "/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth" 
state_dict = torch.load(local_weights_path, map_location="cpu", weights_only=True)
model.encoder.load_state_dict(state_dict)
model = model.to(device)

print("Model loaded successfully")
print("Encoder parameters:", sum(p.numel() for p in model.encoder.parameters()))
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

print("Frozen parameters:",
      sum(p.numel() for p in model.parameters() if not p.requires_grad))


### Gradient Accumulation with pretrained and forzen encoder

optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=1e-3,
    weight_decay=0,
)

model.train()
optimizer.zero_grad()

accumulate_steps = 2

for step, (batch_images, batch_masks) in enumerate(train_loader):
    batch_images = batch_images.to(device)
    batch_masks = batch_masks.to(device)

    outputs = model(batch_images)
    loss = loss_fn(outputs, batch_masks)

    # Scale loss because gradients are accumulated over 2 minibatches
    loss = loss / accumulate_steps
    loss.backward()

    print(f"Minibatch {step + 1}, scaled loss: {loss.item():.4f}")

    if (step + 1) % accumulate_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        print("Optimizer step")
        break


## Validation loop

from transforms_pytorch import build_val_transform

val_transform = build_val_transform(aug_cfg)

val_dataset = SegmentationDataset(
    data_dir="/scratch0/f89601470/ZenklEtAl2026/val/data",
    labels_dir="/scratch0/f89601470/ZenklEtAl2026/val/labels",
    transform=val_transform,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
)

print("Validation dataset size:", len(val_dataset))

val_images, val_masks = next(iter(val_loader))

print("Images:", val_images.shape, val_images.dtype)
print("Masks:", val_masks.shape, val_masks.dtype)
print("Mask classes:", torch.unique(val_masks))

from torchmetrics.classification import MulticlassJaccardIndex

iou_metric = MulticlassJaccardIndex(
    num_classes=5,
    ignore_index=0,
).to(device)

model.eval()

with torch.no_grad():
    val_images = val_images.to(device)
    val_masks = val_masks.to(device)

    val_outputs = model(val_images)

    val_iou = iou_metric(val_outputs, val_masks)

print("Validation IoU:", val_iou.item())

# complete validation pass
iou_metric.reset()
model.eval()

with torch.no_grad():
    for val_images, val_masks in val_loader:
        val_images = val_images.to(device)
        val_masks = val_masks.to(device)

        val_outputs = model(val_images)
        iou_metric.update(val_outputs, val_masks)

val_iou = iou_metric.compute()

print("Validation IoU:", val_iou.item())

## class-specific metrics

from metrics import single_metrics

print(single_metrics.SingleMulticlassF1)
print(single_metrics.SingleMulticlassJaccardIndex)

metrics = {
    "Leaf_Damage_IoU": single_metrics.SingleMulticlassJaccardIndex(
        class_id=2,
        num_classes=5,
        average="none",
    ).to(device),

    "insect_damage_IoU": single_metrics.SingleMulticlassJaccardIndex(
        class_id=3,
        num_classes=5,
        average="none",
    ).to(device),

    "Powdery_Mildew_IoU": single_metrics.SingleMulticlassJaccardIndex(
        class_id=4,
        num_classes=5,
        average="none",
    ).to(device),

    "Leaf_Damage_F1": single_metrics.SingleMulticlassF1(
        class_id=2,
        num_classes=5,
        average="none",
    ).to(device),

    "insect_damage_F1": single_metrics.SingleMulticlassF1(
        class_id=3,
        num_classes=5,
        average="none",
    ).to(device),

    "Powdery_Mildew_F1": single_metrics.SingleMulticlassF1(
        class_id=4,
        num_classes=5,
        average="none",
    ).to(device),
}

iou_metric.reset()

for metric in metrics.values():
    metric.reset()


# all metrics

model.eval()

iou_metric.reset()
for metric in metrics.values():
    metric.reset()

with torch.no_grad():
    for val_images, val_masks in val_loader:
        val_images = val_images.to(device)
        val_masks = val_masks.to(device)

        val_outputs = model(val_images)

        iou_metric.update(val_outputs, val_masks)

        for metric in metrics.values():
            metric.update(val_outputs, val_masks)

results = {
    "IoU": iou_metric.compute().item(),
}

for name, metric in metrics.items():
    value = metric.compute()
    results[name] = value.item() if value.numel() == 1 else value

print(results)