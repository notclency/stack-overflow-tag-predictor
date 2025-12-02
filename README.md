# 🏷️ Stack Overflow Tag Predictor (NLP)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![NLP](https://img.shields.io/badge/NLP-BiLSTM%20%2B%20CNN-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

A hybrid deep learning model that predicts the programming language tag for a given Stack Overflow question title. The model combines **Bi-LSTMs** (for semantic context), **Character-level CNNs** (for handling typos and unknown words), and an **Attention Mechanism** (to focus on key technical terms).

**[🌐 View Live Inference Demo](https://clencytabe.vercel.app/projects/nlp-tag-predictor)**

---

## 🚀 Features

### 1. Hybrid Architecture
The model uses a multi-input strategy to capture different aspects of the text:
- **Bi-LSTM (Bidirectional Long Short-Term Memory):** Captures the semantic meaning and word order of the question.
- **Character CNN:** Extracts sub-word features (e.g., recognizing "Py" in "Python" or handling "NullPtrExcp").
- **Attention Layer:** Automatically learns to "pay attention" to specific keywords (like "pandas", "spring", "swift") while ignoring stopwords ("how", "to", "the").

### 2. Live Inference UI
- Interactive React dashboard that simulates the model's prediction logic.
- Real-time **Attention Heatmap** visualization showing which words triggered the prediction.
- Handles common programming queries for 10 major languages/frameworks.

### 3. Robust Training Pipeline
- **Class Weighting:** Implemented to handle the severe imbalance between popular tags (JavaScript) vs smaller tags (iOS).
- **Cosine Annealing:** Learning rate scheduler for better convergence.
- **AdamW Optimizer:** Used with decoupled weight decay for improved generalization.

---

## 📊 Model Performance

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Test Accuracy** | **77.2%** | Achieved after 50 epochs on unseen test data. |
| **F1-Score (Macro)** | **0.74** | Balanced precision/recall across all 10 classes. |
| **Parameters** | **5.1M** | Total trainable parameters (Embeddings + LSTM + CNN). |

### Supported Tags (Classes)
`python` • `javascript` • `java` • `c#` • `android` • `php` • `c++` • `ios` • `html` • `sql`

---

## 🛠️ Tech Stack

### Deep Learning (Backend Logic)
- **Framework:** PyTorch
- **Layers:** `nn.LSTM`, `nn.Conv1d`, `nn.Embedding`
- **Tokenizer:** TorchText (BasicEnglish)
- **Dataset:** Google BigQuery (Stack Overflow Public Dataset)

### Frontend (Demo)
- **Framework:** React + Vite
- **Styling:** Tailwind CSS + Shadcn UI
- **Animations:** Framer Motion
- **Icons:** Lucide React

---

## 🧠 Architecture Overview

The model processes input text through two parallel paths:

1. **Word Path:** 
   - Input: `[Batch, Seq_Len]`
   - Layer: Pre-trained GloVe Embeddings → Bi-LSTM
   - Output: Contextual Vectors

2. **Character Path:**
   - Input: `[Batch, Seq_Len, Word_Len]`
   - Layer: Char Embedding → Conv1d → MaxPool
   - Output: Morphological Features

**Fusion:** The outputs are concatenated and passed through a final **Attention Layer** before the Softmax classifier.

---

## 📸 Screenshots

| Inference UI | Training Metrics |
|:---:|:---:|
| <img src="https://via.placeholder.com/400x250?text=Live+Prediction" alt="Inference UI" width="400"/> | <img src="https://via.placeholder.com/400x250?text=Accuracy+Charts" alt="Training Metrics" width="400"/> |

---

## 💻 Running Locally

### Prerequisites
- Python 3.8+
- PyTorch (with CUDA if training)
- Node.js (for frontend)

### 1. Train the Model (Python)
```bash
# Clone the repo
git clone https://github.com/your-username/stack-overflow-tag-predictor.git
cd stack-overflow-tag-predictor

# Install Python deps
pip install -r requirements.txt

# Start training script
python train.py --epochs 20 --batch_size 64
