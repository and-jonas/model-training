import torch
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

from torch.nn import CrossEntropyLoss
from torchmetrics.classification import (
    MulticlassJaccardIndex,
    MulticlassF1Score,
)


class SegmentationModel(pl.LightningModule):
    def __init__(
        self,
        learning_rate,
        weight_decay,
        strategy, 
        num_classes,
        class_names,
        metric_classes,
        class_weights=None,
        encoder_name="mit_b3",
        encoder_weights_path=None,
        ignore_index=None,
    ):
        super().__init__()

        self.save_hyperparameters()

        # ---------------------------------------------------------
        # Model
        # ---------------------------------------------------------
        self.model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
        )

        # ---------------------------------------------------------
        # Optional local encoder weights
        # ---------------------------------------------------------
        if encoder_weights_path is not None:
            state_dict = torch.load(
                encoder_weights_path,
                map_location="cpu",
                weights_only=True,
            )

            self.model.encoder.load_state_dict(state_dict)

        # ---------------------------------------------------------
        # Strategy
        # ---------------------------------------------------------
        
        if self.hparams.strategy == "freeze":
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        elif self.hparams.strategy in ["finetune", "scratch"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        else:
            raise ValueError(f"Unknown strategy: {self.hparams.strategy}")

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            weights = weights / weights.sum()
        else:
            weights = None

        self.loss_fn = CrossEntropyLoss(
            weight=weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )

        # ---------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------
        self.val_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        self.val_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index,
        )

        self.val_class_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="none",
            ignore_index=ignore_index,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.size(0),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch

        logits = self(images)
        loss = self.loss_fn(logits, masks)

        predictions = torch.argmax(logits, dim=1)

        self.val_iou.update(predictions, masks)
        self.val_class_iou.update(predictions, masks)
        self.val_class_f1.update(predictions, masks)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=images.size(0),
        )

        return loss

    def on_validation_epoch_end(self):
        iou = self.val_iou.compute()
        class_iou = self.val_class_iou.compute()
        class_f1 = self.val_class_f1.compute()

        self.log(
            "IoU",
            iou,
            prog_bar=True,
            logger=True,
        )

        # Log all classes automatically

        for class_name in self.hparams.metric_classes:
            class_idx = self.hparams.class_names.index(class_name)

            self.log(
                f"{class_name}_IoU",
                class_iou[class_idx],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                f"{class_name}_F1",
                class_f1[class_idx],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer