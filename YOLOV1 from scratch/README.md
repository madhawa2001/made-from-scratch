# YOLO Loss Function

## Overview

This repository implements the loss function for YOLOv1 (You Only Look Once), a state-of-the-art, real-time object detection system. The YOLO model frames object detection as a single regression problem, predicting bounding boxes and class probabilities directly from full images in one evaluation.

## Loss Function Details

The loss function is a multi-part function that simultaneously optimizes:
- Bounding box coordinates
- Bounding box dimensions
- Object confidence
- Classification accuracy

### Mathematical Definition

The complete loss function is defined as:

```
L = λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²]
  + λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]
  + ∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
  + λnoobj∑S²i=0∑Bj=0 ✶ⁿᵒᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
  + ∑S²i=0 ✶ᵒᵇʲᵢ ∑c∈classes(pᵢ(c) - p̂ᵢ(c))²
```

### Components Explained

1. **Bounding Box Position Error**: Penalizes errors in the predicted center coordinates (x,y) of bounding boxes.
   ```
   λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²]
   ```

2. **Bounding Box Size Error**: Penalizes errors in predicted width and height of bounding boxes. Square roots are applied to reduce the impact of errors in larger boxes.
   ```
   λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]
   ```

3. **Object Confidence Error**: Penalizes errors in confidence scores for boxes that should contain objects.
   ```
   ∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
   ```

4. **No-Object Confidence Error**: Penalizes confidence scores for boxes that should not contain objects.
   ```
   λnoobj∑S²i=0∑Bj=0 ✶ⁿᵒᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²
   ```

5. **Classification Error**: Penalizes errors in class probability predictions for cells containing objects.
   ```
   ∑S²i=0 ✶ᵒᵇʲᵢ ∑c∈classes(pᵢ(c) - p̂ᵢ(c))²
   ```

### Parameters and Notation

- **S²**: The image is divided into an S×S grid of cells
- **B**: Each cell predicts B bounding boxes
- **λcoord**: Weight parameter (typically >1) that increases the importance of localization errors
- **λnoobj**: Weight parameter (typically <1) that decreases the importance of confidence errors for cells without objects
- **✶ᵒᵇʲᵢⱼ**: Indicator function that equals 1 if the jth bounding box in cell i is responsible for detecting an object
- **✶ⁿᵒᵒᵇʲᵢⱼ**: Indicator for boxes not responsible for object detection
- **✶ᵒᵇʲᵢ**: Indicator that equals 1 if an object appears in cell i
- **(xᵢ, yᵢ)**: Ground truth coordinates of box center relative to grid cell
- **(x̂ᵢ, ŷᵢ)**: Predicted coordinates of box center
- **wᵢ, hᵢ**: Ground truth width and height of the box (relative to the image dimensions)
- **ŵᵢ, ĥᵢ**: Predicted width and height
- **Cᵢ**: Ground truth confidence score (IoU between the predicted box and any ground truth box)
- **Ĉᵢ**: Predicted confidence score
- **pᵢ(c)**: Ground truth probability of class c in cell i
- **p̂ᵢ(c)**: Predicted probability of class c in cell i

## Implementation Details

The loss function is designed to balance several competing objectives:

1. **Localization accuracy**: The λcoord parameter (typically set to 5) increases the weight of coordinate and size errors to improve localization accuracy.

2. **Confidence prediction**: The model predicts confidence scores representing the IoU (Intersection over Union) between the predicted box and ground truth.

3. **No-object confidence suppression**: The λnoobj parameter (typically set to 0.5) reduces the penalty for confidence predictions in cells with no objects, addressing the class imbalance problem.

4. **Classification accuracy**: The model predicts class probabilities conditioned on the cell containing an object.

## Usage

```python
# Example usage
loss = YOLOLoss(S=7, B=2, lambda_coord=5.0, lambda_noobj=0.5)
output = model(images)
loss_value = loss(output, targets)
```

## References

- Original YOLO paper: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
