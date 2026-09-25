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
        class_weights,
        num_classes=5,
        encoder_name="mit_b3",
        encoder_weights_path="/scratch0/f89601470/torch/hub/checkpoints/mit_b3.pth",
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

        # Load ImageNet-pretrained encoder weights locally.
        # This avoids downloading weights on the compute node.
        state_dict = torch.load(
            encoder_weights_path,
            map_location="cpu",
            weights_only=True,
        )

        self.model.encoder.load_state_dict(state_dict)

        # Freeze encoder, equivalent to the old "strategy: freeze".
        for param in self.model.encoder.parameters():
            param.requires_grad = False

        # ---------------------------------------------------------
        # Loss
        # ---------------------------------------------------------
        weights = torch.tensor(
            class_weights,
            dtype=torch.float32,
        )

        # Preserve the weighting behaviour of the previous script.
        weights = weights / weights.sum()

        self.loss_fn = CrossEntropyLoss(weight=weights)

        # ---------------------------------------------------------
        # Validation metrics
        # ---------------------------------------------------------

        # Overall IoU, ignoring background.
        self.val_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            ignore_index=0,
        )

        # Per-class IoU.
        self.val_class_iou = MulticlassJaccardIndex(
            num_classes=num_classes,
            average="none",
        )

        # Per-class F1.
        self.val_class_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="none",
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

        # Update TorchMetrics.
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
        # Compute metrics accumulated over the entire validation set.
        iou = self.val_iou.compute()
        class_iou = self.val_class_iou.compute()
        class_f1 = self.val_class_f1.compute()

        # Overall IoU.
        self.log(
            "IoU",
            iou,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Class-specific IoU.
        self.log(
            "Leaf_Damage_IoU",
            class_iou[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "insect_damage_IoU",
            class_iou[3],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "Powdery_Mildew_IoU",
            class_iou[4],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Class-specific F1.
        self.log(
            "Leaf_Damage_F1",
            class_f1[2],
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.log(
            "insect_damage_F1",
            class_f1[3],
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        self.log(
            "Powdery_Mildew_F1",
            class_f1[4],
            on_step=False,
            on_epoch=True,
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