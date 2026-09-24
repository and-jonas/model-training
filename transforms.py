from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple
from flash.core.data.io.input import DataKeys
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import AlbumentationsAdapter, ApplyToKeys
from flash.core.utilities.imports import _ALBUMENTATIONS_AVAILABLE, _TORCHVISION_AVAILABLE, requires
from flash.image.segmentation.input_transform import prepare_target, permute_target, remove_extra_dimensions, SemanticSegmentationInputTransform
if _ALBUMENTATIONS_AVAILABLE:
    import albumentations as alb
else:
    alb = None
if _TORCHVISION_AVAILABLE:
    from torchvision import transforms as T

@dataclass
class SegmentationAugmentation(SemanticSegmentationInputTransform):

    def __init__(self, config):
        t_c = config.train
        self.train_t = []
        if t_c.hflip:
            self.train_t.append(alb.HorizontalFlip(p=0.5))
        if t_c.vflip:
            self.train_t.append(alb.VerticalFlip(p=0.5))
        if t_c.resize != 0:
            self.train_t.append(alb.Resize(height=(t_c.resize), width=(t_c.resize), p=1))
        if t_c.random_scale != 0:
            self.train_t.append(alb.RandomScale(scale_limit=(0, t_c.random_scale)))
        if t_c.random_rotate != 0:
            self.train_t.append(alb.Rotate(limit=(t_c.random_rotate)))
        if t_c.random_crop != 0:
            self.train_t.append(alb.RandomCrop(height=(t_c.random_crop), width=(t_c.random_crop)))
        if t_c.color_jitter.brightness != 0 or t_c.color_jitter.contrast != 0 or t_c.color_jitter.contrast != 0 or t_c.color_jitter.hue != 0:
            self.train_t.append((alb.ColorJitter)(**t_c.color_jitter))
        if t_c.gaussian_blur != 0:
            self.train_t.append(alb.Defocus(radius=(t_c.gaussian_blur)))
        if t_c.gaussian_noise != 0:
            self.train_t.append(alb.GaussNoise(var_limit=(t_c.gaussian_noise)))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.train_t.append(alb.Normalize(mean=mean, std=std))
        t_v = config.val
        self.val_t = [alb.Resize(height=(t_v.resize), width=(t_v.resize)),
         alb.Normalize(mean=mean, std=std)]
        super().__init__()

    @requires("image")
    def train_per_sample_transform(self) -> Callable:
        return T.Compose([
         permute_target,
         AlbumentationsAdapter(self.train_t),
         ApplyToKeys(DataKeys.INPUT, T.ToTensor())])

    @requires("image")
    def per_sample_transform(self) -> Callable:
        return T.Compose([
         permute_target,
         AlbumentationsAdapter(self.val_t),
         ApplyToKeys(DataKeys.INPUT, T.ToTensor())])

    def per_batch_transform(self) -> Callable:
        return T.Compose([prepare_target, remove_extra_dimensions])
