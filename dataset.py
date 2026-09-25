import cv2
import torch
from pathlib import Path
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, labels_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform

        self.image_paths = sorted(self.data_dir.glob("*.png"))

        if not self.image_paths:
            raise RuntimeError(f"No PNG images found in {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / image_path.name

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label not found for {image_path.name}: {label_path}"
            )

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=mask,
            )
            image = transformed["image"]
            mask = transformed["mask"]

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return image, mask