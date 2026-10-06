from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
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

# task-specific configuration
task = "focus"
OPTIMIZATION_CONFIG_PATH = f"config/{task}/optimization/base.yaml"
MODEL_CONFIG_PATH = f"config/{task}/model/base.yaml"
AUGMENTATION_CONFIG_PATH = f"config/{task}/data_augmentation/base.yaml"
DATASET_CONFIG_PATH = f"config/{task}/dataset/base.yaml"
optimization_cfg = OmegaConf.load(OPTIMIZATION_CONFIG_PATH)
model_cfg = OmegaConf.load(MODEL_CONFIG_PATH)
augmentation_cfg = OmegaConf.load(AUGMENTATION_CONFIG_PATH)
dataset_cfg = OmegaConf.load(DATASET_CONFIG_PATH)

# Datasets
train_transform = build_train_transform(augmentation_cfg)
val_transform = build_val_transform(augmentation_cfg)
train_dataset = SegmentationDataset(
    data_dir=dataset_cfg.train.data,
    labels_dir=dataset_cfg.train.labels,
    transform=train_transform,
)
val_dataset = SegmentationDataset(
    data_dir=dataset_cfg.val.data,
    labels_dir=dataset_cfg.val.labels,
    transform=val_transform,
)

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=optimization_cfg.batch_size,
    shuffle=True,
    num_workers=optimization_cfg.n_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=optimization_cfg.batch_size,
    shuffle=False,
    num_workers=optimization_cfg.n_workers,
    pin_memory=True,
)

# Model
model = SegmentationModel(
    learning_rate=float(optimization_cfg.learning_rate),
    weight_decay=float(optimization_cfg.weight_decay),
    strategy=str(optimization_cfg.strategy),
    class_weights=list(optimization_cfg.class_weights),
    num_classes=int(model_cfg.n_classes),
    class_names=model_cfg.class_names,
    metric_classes=model_cfg.metric_classes,
    encoder_name=model_cfg.backbone,
    encoder_weights_path=model_cfg.encoder_weights_path,
)

# Parameter summary
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

# TensorBoard
logger = TensorBoardLogger(
    save_dir="./loggers",
    name=f"{task}_mitb3_fpn",
)

# Callbacks
lr_monitor = LearningRateMonitor(
    logging_interval="epoch",
)
checkpoint = ModelCheckpoint(
    dirpath=Path(logger.log_dir) / "checkpoints",
    filename="epoch{epoch:02d}-valIoU{IoU:.4f}",
    monitor="IoU",
    mode="max",
    save_top_k=1,
    save_last=True,
)
callbacks = [lr_monitor, checkpoint]
if optimization_cfg.early_stopping.enabled:
    early_stopping = EarlyStopping(
        monitor=optimization_cfg.early_stopping.monitor,
        mode=optimization_cfg.early_stopping.mode,
        patience=optimization_cfg.early_stopping.patience,
        min_delta=optimization_cfg.early_stopping.min_delta,
    )
    callbacks.append(early_stopping)

# train
trainer = pl.Trainer(
    max_epochs=int(optimization_cfg.epochs),
    accelerator="gpu",
    devices=[int(optimization_cfg.device)],
    accumulate_grad_batches=optimization_cfg.accumulate_grad_batches,
    logger=logger,
    callbacks=callbacks,
    log_every_n_steps=10,
    enable_checkpointing=True,
    num_sanity_val_steps=0
)
trainer.fit(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)