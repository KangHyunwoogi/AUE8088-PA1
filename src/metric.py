from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        # 상태 변수 초기화
        self.add_state('true_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_positives', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('false_negatives', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # 예측값에서 가장 높은 확신을 가진 인덱스를 추출
        preds = torch.argmax(preds, dim=1)

        # 예측값과 타겟의 모양이 같은지 확인
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")

        # True Positives, False Positives, False Negatives 계산
        self.true_positives += torch.sum((preds == target) & (target == 1))
        self.false_positives += torch.sum((preds != target) & (preds == 1))
        self.false_negatives += torch.sum((preds != target) & (target == 1))

    def compute(self):
        # 정밀도와 재현율 계산
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-6)  # 0으로 나누는 것을 방지하기 위해 작은 수를 추가
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-6)

        # F1 점수 계산
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)  # 0으로 나누는 것을 방지
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
