from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset import SegmentationDataset
from transforms_pytorch import (
    build_train_transform,
    build_val_transform,
)
from lightning_model import SegmentationModel


# ================================================================
# Paths
# ================================================================

CONFIG_PATH = "config/symptoms/optimization/base.yaml"
MODEL_CONFIG_PATH = "config/symptoms/model/base.yaml"
AUGMENTATION_CONFIG_PATH = "config/symptoms/data_augmentation/base.yaml"
DATASET_CONFIG_PATH = "config/symptoms/dataset/base.yaml"

TRAIN_DATA = "/scratch0/f89601470/ZenklEtAl2026/train/data"
TRAIN_LABELS = "/scratch0/f89601470/ZenklEtAl2026/train/labels"

VAL_DATA = "/scratch0/f89601470/ZenklEtAl2026/val/data"
VAL_LABELS = "/scratch0/f89601470/ZenklEtAl2026/val/labels"

ENCODER_WEIGHTS = (
    "/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth"
)


# ================================================================
# Load configuration
# ================================================================

base_cfg = OmegaConf.load(CONFIG_PATH)
model_cfg = OmegaConf.load(MODEL_CONFIG_PATH)
augmentation_cfg = OmegaConf.load(AUGMENTATION_CONFIG_PATH)
dataset_cfg = OmegaConf.load(DATASET_CONFIG_PATH)


# ================================================================
# Print configuration
# ================================================================

print("\nConfiguration")
print("-------------")
print(f"Optimizer:       {base_cfg.optimizer.name}")
print(f"Epochs:          {base_cfg.epochs}")
print(f"Learning rate:   {base_cfg.learning_rate}")
print(f"Weight decay:    {base_cfg.weight_decay}")
print(f"Batch size:      {base_cfg.batchsize}")
print(f"Minibatch size:  {base_cfg.minibatch_size}")
print(f"Strategy:        {base_cfg.strategy}")
print(f"Workers:         {base_cfg.n_workers}")
print(f"Backbone:        {model_cfg.backbone}")
print(f"Head:            {model_cfg.head}")
print(f"Classes:         {model_cfg.n_classes}")
print(f"Dataset:         {dataset_cfg.dataset_name}")


# ================================================================
# Effective batch size
# ================================================================

accumulate_grad_batches = int(
    base_cfg.batchsize / base_cfg.minibatch_size
)

if (
    base_cfg.batchsize % base_cfg.minibatch_size != 0
):
    raise ValueError(
        "batchsize must be divisible by minibatch_size"
    )

print(
    f"Gradient accumulation: {accumulate_grad_batches}"
)


# ================================================================
# Datasets
# ================================================================

train_transform = build_train_transform(
    augmentation_cfg
)

val_transform = build_val_transform(
    augmentation_cfg
)

train_dataset = SegmentationDataset(
    data_dir=TRAIN_DATA,
    labels_dir=TRAIN_LABELS,
    transform=train_transform,
)

val_dataset = SegmentationDataset(
    data_dir=VAL_DATA,
    labels_dir=VAL_LABELS,
    transform=val_transform,
)

print(f"\nTraining images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")


# ================================================================
# DataLoaders
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=base_cfg.minibatch_size,
    shuffle=True,
    num_workers=base_cfg.n_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=base_cfg.minibatch_size,
    shuffle=False,
    num_workers=base_cfg.n_workers,
    pin_memory=True,
)


# ================================================================
# Model
# ================================================================

model = SegmentationModel(
    learning_rate=float(base_cfg.learning_rate),
    weight_decay=float(base_cfg.weight_decay),
    class_weights=list(base_cfg.class_weights),
    num_classes=int(model_cfg.n_classes),
    encoder_name=model_cfg.backbone,
    encoder_weights_path=ENCODER_WEIGHTS,
)


# ================================================================
# Parameter summary
# ================================================================

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nModel")
print("-----")
print(f"Total parameters:     {total_parameters:,}")
print(f"Trainable parameters: {trainable_parameters:,}")


# ================================================================
# TensorBoard
# ================================================================

logger = TensorBoardLogger(
    save_dir="./loggers",
    name="ZenklEtAl2026_mitb3_fpn",
)

print(f"\nTensorBoard log directory:")
print(logger.log_dir)


# ================================================================
# Callbacks
# ================================================================

lr_monitor = LearningRateMonitor(
    logging_interval="epoch",
)

checkpoint_callback = ModelCheckpoint(
    dirpath=Path(logger.log_dir) / "checkpoints",
    filename="epoch{epoch:02d}-valIoU{IoU:.4f}",
    monitor="IoU",
    mode="max",
    save_top_k=1,
    save_last=True,
)


# ================================================================
# Trainer
# ================================================================

trainer = pl.Trainer(
    max_epochs=int(base_cfg.epochs),

    accelerator="gpu",
    devices=[int(base_cfg.device)],

    accumulate_grad_batches=accumulate_grad_batches,

    logger=logger,

    callbacks=[
        lr_monitor,
        checkpoint_callback,
    ],

    log_every_n_steps=10,

    enable_checkpointing=True,

    # Do not run an extra validation sanity-check epoch before
    # training. This keeps the first actual validation epoch aligned
    # with the previous training workflow.
    num_sanity_val_steps=0,
)


# ================================================================
# Training
# ================================================================

trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)


# ================================================================
# Final information
# ================================================================

print("\nTraining finished.")

print(f"Best checkpoint:")
print(checkpoint_callback.best_model_path)

print(f"Best IoU:")
print(checkpoint_callback.best_model_score)

print(f"\nTensorBoard:")
print(f"tensorboard --logdir {logger.log_dir}")