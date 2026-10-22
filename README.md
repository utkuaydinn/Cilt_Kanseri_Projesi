# 🔬 Skin Cancer (Melanoma) Detection & Classification via Deep Learning

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Architecture-EfficientNet--V2--S%20%7C%20ResNet--50-blue.svg)](https://arxiv.org/abs/2104.00298)
[![Loss Function](https://img.shields.io/badge/Loss-Focal%20Loss%20(%CE%B3%3D2.0%2C%20%CE%B1%3D0.85)-darkgreen.svg)](https://arxiv.org/abs/1708.02002)
[![Hardware](https://img.shields.io/badge/Hardware-Apple%20Silicon%20(MPS)-silver.svg?logo=apple)](https://developer.apple.com/metal/pytorch/)
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-purple.svg)](https://arxiv.org/abs/1610.02391)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end medical computer vision system that classifies dermoscopic skin lesions into **Benign** vs. **Malignant (Melanoma)** using **EfficientNet-V2-S** and **ResNet-50**, engineered specifically to overcome extreme class imbalance via **Focal Loss** and provide interpretable diagnoses via **Grad-CAM heatmaps**.

---

## 📌 Project Highlights & Clinical Motivation

Early diagnosis of malignant melanoma increases the 5-year survival rate to over 99%. However, machine learning in dermatology faces a critical bottleneck: **Extreme Class Imbalance** (~98% benign vs ~2% malignant). 

A naive model predicting "benign" for every sample achieves **98% overall accuracy** while failing to detect a single cancer patient. This project addresses the **Accuracy Paradox** in clinical AI by:
1. Penalizing majority classes via **Focal Loss ($\gamma=2.0, \alpha=0.85$)** and **WeightedRandomSampler**.
2. Comparing **ResNet-50** against modern **EfficientNet-V2-S** (Compound Scaling & Fused-MBConv).
3. Optimizing training on **Apple Silicon (M3 MPS)** for zero thermal throttling and unified memory efficiency.
4. Implementing **Grad-CAM (Gradient-weighted Class Activation Mapping)** for physician explainability.
5. Evaluating decision boundaries across multiple clinical safety thresholds (%50, %75, %85).

---

## 📊 Architectural Showdown: ResNet-50 vs. EfficientNet-V2-S

| Specification / Metric | ResNet-50 (Residual Networks) | EfficientNet-V2-S (Compound Scaling) |
| :--- | :---: | :---: |
| **Parameters** | 25.5 Million | **21.5 Million (~20% smaller)** |
| **Model Size** | ~98 MB | **~81 MB** |
| **Peak Validation AUC** | **0.9000** (Epoch 5) | **0.8940** (Epoch 1) |
| **Architectural Core** | 50 Conv Layers + Skip Connections | Fused-MBConv + Progressive NAS |
| **Thermal & Hardware Fit** | High RAM & GPU overhead | **Optimized for M3 Apple Silicon (MPS)** |
| **Production Recommendation** | Baseline benchmark | **Selected Final Production Model** |

### 30-Epoch Training & Overfitting Dynamics
Both models were trained across 30 epochs with `ReduceLROnPlateau` scheduling. As shown below, early stopping at Epoch 1 (EfficientNet) and Epoch 5 (ResNet) yielded optimal generalization before validation loss degraded:

![Model Comparison](./model_karsilastirma.png)

---

## ⚖️ The Accuracy Paradox & Clinical Threshold Tuning

In screening for life-threatening conditions, a **False Negative (FN - missing cancer)** is catastrophic, whereas a **False Positive (FP - unnecessary biopsy)** is an acceptable trade-off.

We benchmarked the trained EfficientNet-V2-S across **6,626 validated test images**:

| Decision Threshold | Overall Accuracy | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Clinical Impact |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **50% (Recommended Screening)** | **87.81%** | **87** | 778 | **30** | **Life-Saving:** Minimizes missed cancers (Recall: **74.36%**). |
| **75% (Balanced)** | **95.71%** | 39 | 206 | 78 | **Balanced:** Fewer false alarms, but 78 cancers missed. |
| **85% (Conservative / High Acc)** | **97.75%** | 25 | 57 | **92** | **Dangerous:** High 98% accuracy, but misses 80% of cancers! |

### Confusion Matrices Across Safety Thresholds

| 50% Threshold (Screening Mode) | 75% Threshold (Balanced Mode) | 85% Threshold (Conservative Mode) |
| :---: | :---: | :---: |
| ![CM 50%](./confusion_matrix_50.png) | ![CM 75%](./confusion_matrix_75.png) | ![CM 85%](./confusion_matrix_85.png) |

### ROC-AUC Curve
The model exhibits strong discriminative ability with a high Area Under the Curve (AUC):

![ROC Curve](./roc_curve.png)

---

## 🧠 Explainable AI (XAI) with Grad-CAM

To ensure clinical trustworthiness, **Grad-CAM** computes the gradients of the score for the malignant class with respect to the final convolutional feature maps. This highlights the exact visual features (asymmetric borders, color variegation) driving the prediction.

---

## 📁 Repository Structure

```
.
├── app.py                      # Interactive Gradio web application with Grad-CAM
├── main.py                     # Primary model training pipeline
├── model.py                    # EfficientNet-V2-S & ResNet-50 model definitions
├── trainer.py                  # Training loops, Focal Loss & optimizer configuration
├── dataset.py                  # ISIC dataset loader & data augmentations
├── evaluate.py                 # Multi-threshold evaluation & metrics computation
├── compare_models.py           # Epoch-by-epoch loss & AUC comparison visualizer
├── predict.py                  # Single-image inference CLI
├── utils.py                    # Device selection (MPS/CUDA/CPU) & helpers
├── requirements.txt            # Python dependencies
├── Mimari_Karsilastirma_Raporu.md # In-depth technical architecture report
├── Egitim_Sonuclari_Raporu.md  # Detailed epoch results & confusion matrix logs
└── *.png                       # High-resolution ROC curves & confusion matrices
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/utkuaydinn/Cilt_Kanseri_Projesi.git
cd Cilt_Kanseri_Projesi
```

### 2. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Interactive Decision Support App
```bash
python app.py
```
Open `http://localhost:7860` in your browser to test lesion images, adjust sensitivity thresholds, and view Grad-CAM heatmaps.

### 4. Train Models
```bash
# Train EfficientNet-V2-S on Apple Silicon MPS
python main.py --model efficientnet --epochs 30 --batch_size 32

# Train ResNet-50
python main.py --model resnet --epochs 30 --batch_size 32
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
