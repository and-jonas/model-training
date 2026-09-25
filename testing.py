import fiftyone as fo
from dataset import FiftyOneSegmentationDataset

dataset = fo.load_dataset("ZenklEtAl2026-train")

torch_dataset = FiftyOneSegmentationDataset(dataset)

print("Length:", len(torch_dataset))

image, mask = torch_dataset[0]

print("Image:")
print("  shape:", image.shape)
print("  dtype:", image.dtype)
print("  min:", image.min())
print("  max:", image.max())

print("\nMask:")
print("  shape:", mask.shape)
print("  dtype:", mask.dtype)
print("  unique:", mask.unique())