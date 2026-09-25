import os
from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from dataset import SegmentationDataset
from transforms_pytorch import build_train_transform, build_val_transform
import metrics.single_metrics as single_metrics

from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

cfg = OmegaConf.load("config/symptoms/optimization/base.yaml")
aug_cfg = OmegaConf.load("config/symptoms/data_augmentation/base.yaml")

device = torch.device(
    f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu"
)

dataset_name = "ZenklEtAl2026"
dataset_root = Path(os.environ["SCRATCH"]) / dataset_name

print("Device:", device)
print("Dataset:", dataset_root)


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------

train_transform = build_train_transform(aug_cfg)
val_transform = build_val_transform(aug_cfg)

train_dataset = SegmentationDataset(
    data_dir=dataset_root / "train" / "data",
    labels_dir=dataset_root / "train" / "labels",
    transform=train_transform,
)

val_dataset = SegmentationDataset(
    data_dir=dataset_root / "val" / "data",
    labels_dir=dataset_root / "val" / "labels",
    transform=val_transform,
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ---------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.minibatch_size,
    shuffle=True,
    num_workers=cfg.n_workers,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.minibatch_size,
    shuffle=False,
    num_workers=cfg.n_workers,
)


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

model = smp.FPN(
    encoder_name="mit_b3",
    encoder_weights=None,
    in_channels=3,
    classes=5,
)

# Load locally downloaded ImageNet weights
local_weights_path = "/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth"

state_dict = torch.load(
    local_weights_path,
    map_location="cpu",
    weights_only=True,
)

model.encoder.load_state_dict(state_dict)

# Freeze ImageNet-pretrained encoder
for param in model.encoder.parameters():
    param.requires_grad = False

model = model.to(device)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

class_weights = torch.tensor(
    list(cfg.class_weights),
    dtype=torch.float32,
    device=device,
)

class_weights = class_weights / class_weights.sum()

loss_fn = CrossEntropyLoss(weight=class_weights)


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

iou_metric = MulticlassJaccardIndex(
    num_classes=5,
    ignore_index=0,
).to(device)

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


# ---------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------

accumulate_steps = int(
    cfg.batchsize / cfg.minibatch_size
)

print("Effective batch size:", cfg.batchsize)
print("Minibatch size:", cfg.minibatch_size)
print("Accumulation steps:", accumulate_steps)


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

for epoch in range(cfg.epochs):

    print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

    # -------------------------
    # Training
    # -------------------------
    model.train()
    optimizer.zero_grad()
    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{cfg.epochs} [train]",
        leave=True,
    )

    for step, (images, masks) in enumerate(train_progress):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        train_loss += loss.item()

        loss_for_backward = loss / accumulate_steps
        loss_for_backward.backward()

        if (step + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Running mean rather than current batch loss
        running_train_loss = train_loss / (step + 1)

        train_progress.set_postfix(
            loss=f"{running_train_loss:.4f}"
        )

    train_loss /= len(train_loader)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    iou_metric.reset()
    for metric in metrics.values():
        metric.reset()

    val_loss = 0.0

    val_progress = tqdm(
        val_loader,
        desc=f"Epoch {epoch + 1}/{cfg.epochs} [val]",
        leave=True,
    )

    with torch.no_grad():
        for step, (images, masks) in enumerate(val_progress):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

            # Running mean validation loss
            running_val_loss = val_loss / (step + 1)

            iou_metric.update(outputs, masks)

            for metric in metrics.values():
                metric.update(outputs, masks)

            # Current running metrics
            running_iou = iou_metric.compute().item()

            running_metrics = {}
            for name, metric in metrics.items():
                value = metric.compute()
                running_metrics[name] = value.item()

            val_progress.set_postfix(
                loss=f"{running_val_loss:.4f}",
                IoU=f"{running_iou:.3f}",
                LeafIoU=f"{running_metrics['Leaf_Damage_IoU']:.3f}",
                InsectIoU=f"{running_metrics['insect_damage_IoU']:.3f}",
                PMIoU=f"{running_metrics['Powdery_Mildew_IoU']:.3f}",
            )

    val_loss /= len(val_loader)

    # Final epoch metrics
    results = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "IoU": iou_metric.compute().item(),
    }

    for name, metric in metrics.items():
        value = metric.compute()
        results[name] = value.item()

    print(
        f"Epoch {epoch + 1}/{cfg.epochs} "
        f"- train_loss: {train_loss:.4f} "
        f"- val_loss: {val_loss:.4f} "
        f"- IoU: {results['IoU']:.4f} "
        f"- Leaf IoU: {results['Leaf_Damage_IoU']:.4f} "
        f"- Insect IoU: {results['insect_damage_IoU']:.4f} "
        f"- PM IoU: {results['Powdery_Mildew_IoU']:.4f}"
    )