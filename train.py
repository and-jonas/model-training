import torch
import flash
from flash.image import SemanticSegmentation, SemanticSegmentationData
from torch.nn import CrossEntropyLoss
import fiftyone as fo
import fiftyone.utils.random as four
from itertools import chain
from flash.image.segmentation.output import FiftyOneSegmentationLabelsOutput
import logging
from pathlib import PurePath
import os
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from datetime import datetime
import cv2
from metrics import stb_metrics, single_metrics
from torchmetrics.classification import JaccardIndex, MulticlassJaccardIndex
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig
from transforms import SegmentationAugmentation
# import onnxruntime
import numpy as np
from typing import Tuple
import wandb

def init_datasets():
    # Setup persistent fiftyone datasets with clear version and with train, val split
    dataset_root = os.environ["SCRATCH"]
    images_suffix = 'data'
    labels_suffix = 'labels'
    dataset_names = [
        'ZenklEtAl2026',
    ]
    for dataset_name in dataset_names:
        existing_datasets = fo.list_datasets()
        for split in ['train', 'val']:
            name = dataset_name + '-' + split
            if name in existing_datasets:
                logging.warning("dataset: {} is already present in fiftyone datasets".format(name))
                logging.warning("skipping")
                continue
            dataset =  fo.Dataset.from_dir(name=name,
                dataset_type=fo.types.ImageSegmentationDirectory,
                data_path=str(PurePath(dataset_root, dataset_name, split, images_suffix)),
                labels_path=str(PurePath(dataset_root, dataset_name, split, labels_suffix)),
                shuffle=True,
                seed=42,
                load_masks=True,
                force_grayscale=False,
                )
                
            dataset.persistent=True
            del dataset

@hydra.main(config_path="config/symptoms", config_name="base")
def train(cfg: DictConfig):

    # Run name
    run_name = cfg.run_name + '-' + cfg.model.backbone + '-' + cfg.data_augmentation.name + '-' + cfg.dataset.dataset_name
    entity="ags-digitalisation"
    project='symptoms-EFDv2'
    wandb.init(project=project, entity=entity, name=run_name)
    logger = WandbLogger(save_dir='logs', name=run_name, project=project, entity=entity)
    # Dataset 
    dataset_name = cfg.dataset.dataset_name
    add_predictions_to_dataset = cfg.add_predictions
    # Optimization
    n_epochs = cfg.optimization.epochs
    optimizer = cfg.optimization.optimizer.name
    strategy = cfg.optimization.strategy
    learning_rate = cfg.optimization.learning_rate
    weight_decay = cfg.optimization.weight_decay
    batch_size = cfg.optimization.batchsize
    minibatch_size = cfg.optimization.minibatch_size
    accumulate_size = int(batch_size / minibatch_size)
    device = cfg.optimization.device
    n_workers = cfg.optimization.n_workers
    class_weights = [w for w in cfg.optimization.class_weights]
    weights = torch.tensor(class_weights).long()
    weights = weights / torch.sum(weights)
    # Architecture Configuration
    head = cfg.model.head
    backbone = cfg.model.backbone
    num_classes = cfg.model.n_classes
    # Data Augmentation
    transforms = SegmentationAugmentation(cfg.data_augmentation)
    print('using device: {}'.format(cfg.optimization.device))
    # Check if Dataset exists
    existing_datasets = fo.list_datasets()
    train_name = dataset_name + '-train'
    val_name = dataset_name + '-val'
    if train_name not in existing_datasets:
        raise Exception(f"Dataset not found {train_name}")
    if val_name not in existing_datasets:
        raise Exception(f"Dataset not found {val_name}")
    
    # Convert views to datasets as flash cannot deal with views. This will get hopefully fixed
    train_dataset = fo.load_dataset(train_name)
    val_dataset = fo.load_dataset(val_name)
    # Create Datamodule
    datamodule = SemanticSegmentationData.from_fiftyone(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_field="ground_truth",
        predict_dataset=val_dataset,  # Predict on the val dataset and then filter during inspection
        transform=transforms,
        num_classes=num_classes,
        batch_size=minibatch_size,
        num_workers=n_workers,
    )
    # Build the task
    model = SemanticSegmentation(
        backbone=backbone,
        head=head,
        num_classes=num_classes,
        optimizer=(optimizer, {'weight_decay': weight_decay}),
        loss_fn=CrossEntropyLoss(weight=weights),
        learning_rate=learning_rate,
        metrics={
        'IoU': JaccardIndex(num_classes=num_classes, ignore_index=0),
        'Leaf_Damage_IoU': single_metrics.SingleMulticlassJaccardIndex(class_id=2, num_classes=num_classes, average='none'),
        'insect_damage_IoU': single_metrics.SingleMulticlassJaccardIndex(class_id=3, num_classes=num_classes, average='none'),
        'Powdery_Mildew_IoU': single_metrics.SingleMulticlassJaccardIndex(class_id=4, num_classes=num_classes, average='none'),
        'Leaf_Damage_F1': single_metrics.SingleMulticlassF1(class_id=2, num_classes=num_classes, average='none'),
        'insect_damage_F1': single_metrics.SingleMulticlassF1(class_id=3, num_classes=num_classes, average='none'),
        'Powdery_Mildew_F1': single_metrics.SingleMulticlassF1(class_id=4, num_classes=num_classes, average='none'),
     }
    )
    # Create the trainer and finetune the model
    # early_stopping = EarlyStopping('val_IoU', mode='max', patience=10)
    trainer = flash.Trainer(max_epochs=n_epochs, accelerator='gpu', devices=[device], logger=logger, accumulate_grad_batches=accumulate_size
                            # callbacks=[early_stopping]
                            )
    trainer.finetune(model, datamodule=datamodule, strategy=strategy)
    # Log model predictions
    if add_predictions_to_dataset:
        predictions = trainer.predict(model,
                                    datamodule=datamodule,
                                    output=FiftyOneSegmentationLabelsOutput(return_filepath=True))
        predictions = list(chain.from_iterable(predictions))
        # flatten batches
        # Map filepaths to predictions
        predictions = {p["filepath"]: p["predictions"] for p in predictions}
        # Add predictions to FiftyOne dataset
        val_dataset.set_values(run_name, predictions, key_field="filepath", )
        print("done")
    print("finished training")
    wandb.finish()
def export_torchscript(model_path, export_name, imgsz):
    model = SemanticSegmentation.load_from_checkpoint(model_path)
    with torch.no_grad():
        input_sample = torch.randn((1, 3, imgsz, imgsz))
        model.to_torchscript(method='trace', file_path=export_name, example_inputs=input_sample)
if __name__ == "__main__":
    init_datasets()
    train()
    # export_torchscript(model_path='/LeafOps/models/multirun/2025-01-19/21-00-26/1/logs/organs-reference-benchmark/xkeog5ev/checkpoints/epoch=199-step=7200.ckpt', 
    #                    export_name='org_fpn_mitb2_0.001_v8.torchscript', 
    #                    imgsz=1024,
    #                    )