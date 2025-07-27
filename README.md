# FK-CNN | 从零开始的卷积神经网络

> 🚀 A convolutional neural network implemented from scratch using only NumPy.

手写实现的卷积神经网络，不依赖 PyTorch / TensorFlow，仅使用 NumPy 和 Matplotlib。

---

## 🧠 Project Overview | 项目概览

This project builds a complete convolutional neural network pipeline step by step, starting from dataset loading, to building each layer (convolution, activation, pooling, fully connected, softmax), and assembling them into a functional model.

本项目自底向上构建了完整的卷积神经网络，包括：数据加载、卷积层、激活层、池化层、全连接层与 Softmax 输出层，并支持训练与推理流程。

---

## 🔬 Theory Behind the Code | 核心原理讲解

### 1. Convolutional Layer | 卷积层

Performs feature extraction using sliding filters over input images.

* Each filter has learnable weights and bias.
* Output is computed as element-wise dot product of patch and filter.

**作用**：提取图像的局部特征（边缘、纹理等），保留空间结构。

### 2. Activation (ReLU)

Applies non-linearity: `ReLU(x) = max(0, x)`.

**作用**：引入非线性，提高模型表达能力，抑制负激活。

### 3. Max Pooling

Reduces spatial resolution while retaining dominant features.

* Operates on small regions (e.g., 2x2)
* Keeps the maximum value

**作用**：降低计算量、压缩特征图尺寸、增强不变性（如旋转、平移）。

### 4. Fully Connected Layer

Flattens and connects extracted features to output classes.

**作用**：综合所有特征信息，进行高维空间的线性变换，用于分类。

### 5. Softmax + Cross Entropy

* Softmax normalizes output to probability distribution
* Cross Entropy measures prediction error w\.r.t. true labels

**作用**：将输出映射为概率分布，并计算模型输出与真实标签之间的差异。

---

## 📁 Project Structure | 项目结构

```
handmade_cnn/
├── layers/                # Core layer implementations
│   ├── conv.py           # Conv2D layer
│   ├── activation.py     # ReLU layer
│   ├── pool.py           # MaxPool2D
│   └── fc.py             # Fully connected layer
├── models/
│   └── cnn.py            # CNN model assembly
├── utils/
│   └── logger_util.py    # Logger setup
├── dataset.py            # MNIST loader
├── train.py              # Training loop
├── test.py               # Evaluation script
├── requirements.txt      # Pip dependencies
└── README.md             # This file
```

---

## ⚙️ Installation | 安装方法

### Option : Use pip + venv

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Usage | 使用方法

### Step 1: Load and preprocess data

```bash
python main.py
```

---

## 📌 Features | 特性

* ✅ Handwritten convolution implementation (vectorized)
* ✅ Max pooling with zero-padding and edge handling
* ✅ Fully connected layer with dynamic shape adaptation
* ✅ Softmax + cross-entropy loss with label separation
* ✅ Supports flexible padding, stride, and batch fetching
* ✅ Structured logging & modular architecture

---

## 🔍 Requirements | 环境要求

* Python >= 3.8
* NumPy >= 1.24.0
* Matplotlib >= 3.7.0

---

## 🧑‍💻 Author | 作者

**STEVEN ZHAO**
Handcrafted with ❤️ in 2025-07

---

## 📄 License | 开源协议

GPL License v3.0.

---

本项目适合作为初学者深入理解 CNN 原理的练习项目，也可拓展用于更复杂的图像分类任务。

欢迎 Star / Fork / 提交 Pull Request！

---

*If you'd like a Chinese-only or English-only version, please let me know!*
