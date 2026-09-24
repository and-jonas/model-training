from torchmetrics.classification import JaccardIndex, MulticlassJaccardIndex, MulticlassF1Score
import torch

class SingleMulticlassJaccardIndex(MulticlassJaccardIndex):

    def __init__(self, class_id, **kwargs):
        self.class_id = class_id
        super().__init__(**kwargs)
        
    def compute(self):
        return super().compute()[self.class_id]


class SingleMulticlassF1(MulticlassF1Score):

    def __init__(self, class_id, **kwargs):
        self.class_id = class_id
        super().__init__(**kwargs)
        
    def compute(self):
        return super().compute()[self.class_id]