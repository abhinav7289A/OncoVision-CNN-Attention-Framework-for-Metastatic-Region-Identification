# 🧬 Metastatic Tissue Detection in Digital Pathology Using CNN with Attention Mechanisms

> **A deep learning framework for interpretable detection of metastatic cancer in histopathology images using CNNs and custom self-attention.**  
> *Dataset:* [PatchCamelyon (PCam)](https://patchcamelyon.grand-challenge.org/)

---

## 🚀 Overview

This repository presents a **state-of-the-art deep learning pipeline** for automated detection of **metastatic cancer tissue** in histopathology images, leveraging the **PatchCamelyon (PCam)** benchmark dataset.  
The model integrates a **Convolutional Neural Network (CNN)** backbone with a **custom self-attention block** to dynamically emphasize salient regions (e.g., tumor cells), enhancing both **accuracy** and **interpretability** for binary classification (*metastatic* vs. *non-metastatic*).

---

## 🏆 Key Achievements

| Metric | Value |
|:--|:--:|
| **Test Accuracy** | **83.1%** on 32,768 validation images |
| **F1-Score** | **80.8%** |
| **Precision / Recall** | **93.8% / 70.9%** |
| **Peak Validation Accuracy** | **92.6%** in 9 epochs (early stopping) |
| **Training Dataset** | **222,822** high-resolution (96×96 RGB) patches |
| **Compute** | TPU-accelerated (Kaggle) |
| **Interpretability** | Grad-CAM heatmaps on metastatic regions |

---

## 💡 Applications

### 🏥 Clinical Triage
Automates pre-screening of **whole-slide images (WSIs, ~50K×50K pixels)** from lymph node biopsies to flag high-risk cases, reducing **pathologist workload by up to 30%** and accelerating **breast cancer diagnosis**.

### 🌐 Resource-Limited Settings
Deployable on **edge or cloud environments** for remote pathology, aiding **tumor quantification** and **AI-assisted diagnostics** in underserved regions.

### 🔬 Research and Drug Discovery
Facilitates large-scale analysis for **biomarker discovery** and integrates with **cognitive computing pipelines** for **explainable AI in precision medicine**.

### ⚖️ Ethical AI Integration
Enhances decision support through **interpretable Grad-CAM visualizations**, mitigating bias and supporting **trustworthy AI** in healthcare applications.

---

## 📂 Dataset

**Source:** [PatchCamelyon (PCam)](https://patchcamelyon.grand-challenge.org/)  
Extracted **96×96 RGB patches** from sentinel lymph node WSIs.

| Split | Samples |
|:--|:--:|
| **Train** | 222,822 |
| **Validation** | 39,322 |
| **Test** | 32,768 |

**Labels:**  
- `0` → Non-metastatic  
- `1` → Metastatic  

**Preprocessing:**
- Normalization → [0, 1] range  
- Data augmentation → random flips, ±0.1 brightness, 0.8–1.2 contrast  

---

## 🧠 Model Architecture

### 🔹 CNN Backbone
Three convolutional blocks for hierarchical feature extraction:

| Layer | Output Shape | Description |
|:--|:--|:--|
| Conv1 | 48×48×64 | 3×3 Conv + ReLU + MaxPool(2×2) |
| Conv2 | 24×24×128 | 3×3 Conv + ReLU + MaxPool(2×2) |
| Conv3 | 12×12×256 | 3×3 Conv + ReLU + MaxPool(2×2) |

---

### 🔹 Attention Block

Reshapes features to sequence (144×256) and applies **scaled dot-product self-attention**:

\[
Q = W_Q X, \quad K = W_K X, \quad V = W_V X
\]
\[
A = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right), \quad O = A V W_O
\]

Where:
- \( X \): input sequence  
- \( d_k = 128 \)  
- \( W_{Q,K,V,O} \in \mathbb{R}^{256 \times 128} \)

Followed by **GlobalAveragePooling1D**.

---

### 🔹 Classification Head
- Dense(64, ReLU)  
- Dropout(0.5)  
- Dense(1, Sigmoid)  
- **Total Parameters:** 494,337  
- **L2 Regularization:** \( \lambda = 10^{-4} \)

**Loss Function:**
\[
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i)]
\]

**Optimizer:** Adam (lr = 1e-3)

---

## 📊 Results

### ⚙️ Convergence
- Stable training in **9 epochs**
- Early stopping prevented overfitting

### 🧾 Test Metrics
| Metric | Value |
|:--|:--:|
| Accuracy | 83.1% |
| F1-Score | 80.8% |
| Precision | 93.8% |
| Recall | 70.9% |

---

### 📉 Learning Curves
![Training Curves](assets/training_curves.png)

### 🧩 Confusion Matrix
![Confusion Matrix](assets/confusion_matrix.png)

### 🔥 Grad-CAM Heatmaps
![GradCAM](assets/gradcam_examples.png)

---

## 🏗️ Project Structure

