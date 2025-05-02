from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")
    def update(self, preds, target):
        pred_labels = torch.argmax(preds, dim=1)
        if pred_labels.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {pred_labels.shape}, target {target.shape}")
        for i in range(self.num_classes):
            pred_c = pred_labels == i
            target_c = target == i

            self.tp[i] += (pred_c & target_c).sum()
            self.fp[i] += (pred_c & ~target_c).sum()
            self.fn[i] += (~pred_c & target_c).sum()
            
    def compute(self):
        TP = self.tp.float()
        FP = self.fp.float()
        FN = self.fn.float()
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        pred_labels = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if pred_labels.shape != target.shape:
            raise ValueError(f"Shape mismatch: preds {pred_labels.shape}, target {target.shape}")

        # [TODO] Cound the number of correct prediction
        correct = (pred_labels == target).sum()

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
