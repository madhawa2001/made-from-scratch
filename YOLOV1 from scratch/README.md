## 📉 YOLOv1 Loss Function Explained

YOLOv1 uses a custom loss function that balances localization, confidence, and classification accuracy. It is composed of five parts:

### 🔸 Full Loss Function

L = λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(xᵢ - x̂ᵢ)² + (yᵢ - ŷᵢ)²] + λcoord∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ[(√wᵢ - √ŵᵢ)² + (√hᵢ - √ĥᵢ)²]+ ∑S²i=0∑Bj=0 ✶ᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)  + λnoobj∑S²i=0∑Bj=0 ✶ⁿᵒᵒᵇʲᵢⱼ(Cᵢ - Ĉᵢ)²  +  ∑S²i=0 ✶ᵒᵇʲᵢ ∑c∈classes(pᵢ(c) - p̂ᵢ(c))²

### 🔍 Explanation of Each Term

1. **Localization Loss (x, y)**  
   Measures error in predicted box center coordinates.  
   Applied only if an object is present in the cell and box `j` is responsible.  
   Weighted by `λ_coord` (usually 5).

2. **Localization Loss (w, h)**  
   Measures error in predicted box dimensions, using square roots to reduce sensitivity to large boxes.  
   Also weighted by `λ_coord`.

3. **Confidence Loss (object present)**  
   Penalizes the difference between predicted and actual object confidence (typically the IoU).  
   Applied only when an object is present.

4. **Confidence Loss (no object)**  
   Penalizes boxes that predict high confidence where no object is present.  
   Weighted by `λ_noobj` (usually 0.5).

5. **Classification Loss**  
   Penalizes the error in predicted class probabilities.  
   Only applied to cells that contain objects.

---

### 🔧 Notation

- `S`: Grid size (e.g. 7 → 7x7 grid)
- `B`: Number of bounding boxes per grid cell (e.g. 2)
- `𝟙ᵢⱼ^obj`: 1 if object is present in cell `i`, and box `j` is responsible
- `𝟙ᵢⱼ^noobj`: 1 if no object is present in that box
- `(x, y)`: center coordinates of box relative to grid cell
- `(w, h)`: width and height of box relative to the whole image
- `C`: confidence score (IoU * objectness)
- `p(c)`: class probability for class `c`

---

### 📌 Typical Hyperparameters

```python
lambda_coord = 5
lambda_noobj = 0.5
