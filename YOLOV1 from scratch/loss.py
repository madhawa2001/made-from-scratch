import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
    
    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.B * 5 + self.C)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat((iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)), dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3) # Iobj_i

        # BOX COORDINATES LOSS
        # Penalizes errors in the predicted center coordinates (x,y) of bounding boxes.
        # Penalizes errors in predicted width and height of bounding boxes. Square roots are applied to reduce the impact of errors in larger boxes.
        # λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²] + λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]
        box_predictions = exists_box * (
            best_box *predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25]
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 0:2] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4]+ 1e-6))

        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2),
                            torch.flatten(box_targets, end_dim=-2))
        

        # OBJECT LOSS
        # ∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
        # Penalizes errors in confidence scores for boxes that should contain objects.
        pred_box = (best_box * predictions[..., 25:26] + (1-best_box) * predictions[..., 20:21])

        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]))
        
        # NO OBJECT LOSS
        # λnoobj∑S²i=0∑Bj=0 ✶ⁿᵒᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
        # Penalizes confidence scores for boxes that should not contain objects.
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # CLASS LOSS
        # ∑S²i=0 ✶ᵒᵇʲᵢ ∑c∈classes(pᵢ(c) - p̂ᵢ(c))²
        # Penalizes errors in class probability predictions for cells containing objects.
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], start_dim=2),
            torch.flatten(exists_box * target[..., :20], start_dim=2)
        )


        loss = (
            self.lambda_coord * box_loss  # λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²] + λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]
            + object_loss  # ∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
            + no_object_loss * self.lambda_noobj # λnoobj∑S²i=0∑Bj=0 ✶ⁿᵒᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
            + class_loss # ∑S²i=0 ✶ᵒᵇʲᵢ ∑c∈classes(pᵢ(c) - p̂ᵢ(c))²
        )

        return loss