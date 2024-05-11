from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        # Initialization TP, FP, FN
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # Extract highest confidence score class index 
        preds = torch.argmax(preds, dim=1)

        # Check shape of Prediction and GT
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")

        # Calculate TP, FP, FN for Precision and Recall
        self.true_positives += torch.sum((preds == target) & (target == 1))
        self.false_positives += torch.sum((preds != target) & (preds == 1))
        self.false_negatives += torch.sum((preds != target) & (target == 1))

    def compute(self):
        # Calculate Precision, Recall
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)  # add extremely small value protecting from zero-division error
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)

        # Calculate F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # add extremely small value protecting from zero-division error
        return f1_score
    
    
    
class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)

        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")

        # [TODO] Count the number of correct predictions
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()


    def compute(self):
        return self.correct.float() / self.total.float()
