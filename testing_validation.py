from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig, OmegaConf
import hydra

from dataset import SegmentationDataset
from transforms_pytorch import (
    build_train_transform,
    build_val_transform,
)

from lightning_model import SegmentationModel

import pandas as pd


def run_fold(fold, cfg, metadata, splits, output_dir):

    # Access configurations
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    optimization_cfg = cfg.optimization
    augmentation_cfg = cfg.data_augmentation

    pl.seed_everything(int(cfg.seed) + int(fold), workers=True)

    fold_dir = output_dir / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fold_splits = splits.loc[ splits["fold"] == fold].copy()

    train_ids = set(fold_splits.loc[fold_splits["split"] == "train", "sample_UID"])
    val_ids = set(fold_splits.loc[fold_splits["split"] == "validation", "sample_UID"])

    assert not train_ids & val_ids
    assert train_ids and val_ids

    # get metadata for training and validation samples
    fold_metadata = metadata.merge(
        fold_splits[["sample_UID", "split"]],
        on="sample_UID",
        how="inner",
        validate="one_to_one",
    )
    train_metadata = fold_metadata.loc[fold_metadata["split"] == "train"].copy()
    val_metadata = fold_metadata.loc[fold_metadata["split"] == "validation"].copy()

    print(f"\nFold {fold}")
    print(f"Training images: {len(train_metadata)}")
    print(f"Validation images: {len(val_metadata)}")

        # Datasets
    train_transform = build_train_transform(augmentation_cfg)
    val_transform = build_val_transform(augmentation_cfg)
    train_dataset = SegmentationDataset(
        data_dir=dataset_cfg.data,
        labels_dir=dataset_cfg.labels,
        transform=train_transform,
        sample_ids=train_ids,
    )
    val_dataset = SegmentationDataset(
        data_dir=dataset_cfg.data,
        labels_dir=dataset_cfg.labels,
        transform=val_transform,
        sample_ids=val_ids,
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
        save_dir=str(output_dir),
        name=f"fold_{fold}",
        version="logs",
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
        save_top_k=0,
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
        # devices=[int(optimization_cfg.device)],
        devices=1,
        accumulate_grad_batches=optimization_cfg.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Evaluate on the held-out fold
    fold_metrics = trainer.validate(
        model,
        dataloaders=val_loader,
    )[0]

    # save results
    result = {
        "experiment": cfg.run_name,
        "fold": int(fold),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "epochs": int(optimization_cfg.epochs),
        **{
            key: float(value)
            for key, value in fold_metrics.items()
        },
    }

    pd.DataFrame([result]).to_csv(fold_dir / "results.csv", index=False)

    OmegaConf.save(cfg, fold_dir / "config.yaml", resolve=True)

    return result


@hydra.main(
        version_base=None, 
        config_path="config", 
        config_name="config"
    )
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    task = cfg.task
    experiment = cfg.run_name

    # Access configurations
    dataset_cfg = cfg.dataset
    splits_cfg = cfg.splits

    # make output directory
    output_dir = Path("experiments") / task / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # get samples and their metadata for one fold of an experiment
    metadata = pd.read_csv(Path(dataset_cfg.meta) / "04_sample_metadata_uids_design_alliters.csv", dtype={"plot": "Int64"})
    splits = pd.read_csv(splits_cfg.path)

    # get training and validation sample_UIDs
    splits = splits.loc[(splits["experiment"] == experiment)].copy()

    if splits.empty:
        raise ValueError(f"No splits found for {cfg.splits.experiment}")

    # Determine which folds to run
    available_folds = sorted(splits["fold"].unique())

    if cfg.splits.folds == "all":
        folds = available_folds
    else:
        folds = list(cfg.splits.folds)

    # Optional: select one fold via command line
    if cfg.fold is not None:
        folds = [int(cfg.fold)]

    if not set(folds).issubset(set(available_folds)):
        raise ValueError(
            f"Requested folds {folds} not found. "
            f"Available folds: {available_folds}"
        )

    print(f"Experiment: {experiment}")
    print(f"Folds to run: {folds}")

    # perform evaluation using the pre-defined resampling procedure
    for fold in folds:
        run_fold(fold=fold, cfg=cfg, metadata=metadata, splits=splits, output_dir=output_dir)


if __name__ == "__main__":
    main()