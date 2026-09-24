from torchmetrics import Metric, MeanAbsolutePercentageError, MeanSquaredError
import torch
from torchvision.transforms.functional import gaussian_blur
from skimage.measure import label

def placl(labels: torch.Tensor, lesions_id: int, leaf_id:int):
    # Assuming BxCxHxW
    n_leaf = torch.sum(labels == leaf_id, dim=(1, 2))
    n_lesi = torch.sum(labels == lesions_id, dim=(1, 2))

    # Return Bx1
    return n_lesi / (n_lesi + n_leaf)



# Write a wrapper which changes the class name and selects reduces input to placl before computing
# def placl_metric(Metric, lesions_id, leaf_id, **kwargs):
#     class NewClass(Metric):

#         def __init__():
#             super().__init__(**kwargs)

#         def update(preds, target):
#             labels = torch.argmax(preds, dim=1)
#             preds_placl = placl(labels, lesion_id, leaf_id)
#             target_placl = placl(target, lesion_id, leaf_id)
#             return super().compute(preds_placl, target_placl)

class PlaclMAPE(MeanAbsolutePercentageError):

    def __init__(self, lesions_id, leaf_id, **kwargs):
        self.lesions_id = lesions_id
        self.leaf_id = leaf_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        labels = torch.argmax(preds, dim=1)
        preds_placl = placl(labels, self.lesions_id, self.leaf_id)
        target_placl = placl(target, self.lesions_id, self.leaf_id)
        return super().update(preds_placl, target_placl)

class PlaclMSE(MeanSquaredError):

    def __init__(self, lesions_id, leaf_id, **kwargs):
        self.lesions_id = lesions_id
        self.leaf_id = leaf_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        labels = torch.argmax(preds, dim=1)
        preds_placl = placl(labels, self.lesions_id, self.leaf_id)
        target_placl = placl(target, self.lesions_id, self.leaf_id)
        return super().update(preds_placl, target_placl)


# def erode_to_points(mask: torch.Tensor, keep_edges=False, min_size=5, blur_size = 9, abs_threshold=0.1) -> torch.Tensor:
#     # Assuming BxCxHxW softmaxed probabilities
#     mask = mask.float()

#     mask_smoothed = gaussian_blur(mask, kernel_size=blur_size)
#     window_maxima = torch.nn.functional.max_pool2d(mask_smoothed, min_size, 1, (min_size - 1) // 2)
#     only_maxima = window_maxima == mask_smoothed

#     if abs_threshold:
#         only_maxima[mask_smoothed < abs_threshold] = False

#     if not keep_edges:
#         only_maxima[..., 0, :] = False
#         only_maxima[..., :, 0] = False
#         only_maxima[..., only_maxima.shape[-2] - 1, :] = False
#         only_maxima[..., :, only_maxima.shape[-1] - 1] = False

#     return only_maxima.int()

class PycnidiaMSE(MeanSquaredError):
    def __init__(self, point_id, **kwargs):
        self.point_id = point_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        preds_thresh = preds
        preds_thresh[preds_thresh < 0.5] = 0
        pred_labels = torch.argmax(preds_thresh, dim=1)
        
        preds_points = torch.zeros(preds_thresh.shape[0])
        for i, batch_i in enumerate(pred_labels):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            preds_points[i] = pp


        target_points = torch.zeros(target.shape[0])
        for i, batch_i in enumerate(target):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            target_points[i] = pp

        return super().update(preds_points, target_points)

    
class RustMSE(MeanSquaredError):
    def __init__(self, point_id, **kwargs):
        self.point_id = point_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        preds_thresh = preds
        preds_thresh[preds_thresh < 0.5] = 0
        pred_labels = torch.argmax(preds_thresh, dim=1)
        
        preds_points = torch.zeros(preds_thresh.shape[0])
        for i, batch_i in enumerate(pred_labels):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            preds_points[i] = pp

        target_points = torch.zeros(target.shape[0])
        for i, batch_i in enumerate(target):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            target_points[i] = pp

        # print("preds: {}".format(preds_points))
        # print("target: {}".format(target_points))

        return super().update(preds_points, target_points)
    
class PycnidiaMAPE(MeanAbsolutePercentageError):
    def __init__(self, point_id, **kwargs):
        self.point_id = point_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        preds_thresh = preds
        preds_thresh[preds_thresh < 0.5] = 0
        pred_labels = torch.argmax(preds_thresh, dim=1)
        
        preds_points = torch.zeros(preds_thresh.shape[0])
        for i, batch_i in enumerate(pred_labels):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            preds_points[i] = pp


        target_points = torch.zeros(target.shape[0])
        for i, batch_i in enumerate(target):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            target_points[i] = pp

        return super().update(preds_points, target_points)

    
class RustMAPE(MeanAbsolutePercentageError):
    def __init__(self, point_id, **kwargs):
        self.point_id = point_id
        super().__init__(**kwargs)
        
    def update(self, preds, target):
        preds_thresh = preds
        preds_thresh[preds_thresh < 0.5] = 0
        pred_labels = torch.argmax(preds_thresh, dim=1)
        
        preds_points = torch.zeros(preds_thresh.shape[0])
        for i, batch_i in enumerate(pred_labels):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            preds_points[i] = pp


        target_points = torch.zeros(target.shape[0])
        for i, batch_i in enumerate(target):
            _, pp = label((batch_i == self.point_id).int().cpu().numpy(), return_num=True)
            target_points[i] = pp

        return super().update(preds_points, target_points)