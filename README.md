# Cat vs. Dog Deep Learning Classifier with Semantic OOD Rejection 🐾

An end-to-end computer vision system comparing a **scratch-trained 3-block CNN** against a **fine-tuned MobileNetV2** transfer learning backbone. The production inference pipeline incorporates a **deterministic ImageNet synset index-range gatekeeper** to mitigate Out-of-Distribution (OOD) overconfidence, routing non-pet objects and wild predators to an `"Else"` classification state.

---

## 📌 Project Highlights

- **Empirical Architecture Comparison**: Contrasts a custom convolutional neural network (from random initialization) against an ImageNet-pretrained MobileNetV2 backbone.
- **Two-Phase Transfer Learning**: Frozen feature extraction (Phase 1, 6 epochs) followed by selective top-20 layer unfreezing at a reduced learning rate ($\eta = 10^{-5}$) (Phase 2, 4 epochs).
- **Deterministic OOD Gatekeeping**: Uses contiguous ImageNet-1k synset ranges (`[151, 268]` for 120 domestic canines, `[281, 285]` for domestic felines) to intercept non-pet objects before binary inference.
- **Low-Level Byte Validation**: Automatically purges truncated and non-JFIF byte headers from disk to prevent TensorFlow `DecodeJpeg` runtime segmentation faults.
- **Interactive Web App**: Served via Streamlit with in-memory caching and real-world failure mode demonstrations.

---

## 📊 Benchmark Results

Evaluated across **4,684 held-out validation images** (2,260 Cats, 2,424 Dogs):

| Metric | Custom Baseline CNN | MobileNetV2 (Fine-Tuned) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **85.03%** | **97.44%** | **+12.41%** |
| **Cat Precision / Recall** | 0.81 / 0.90 | 0.98 / 0.96 | +17% Prec / +6% Rec |
| **Dog Precision / Recall** | 0.89 / 0.80 | 0.97 / 0.99 | +8% Prec / +19% Rec |
| **Total Misclassifications** | 701 / 4,684 | **120 / 4,684** | **4.2× Error Reduction** |
| **Convergence Speed** | 10 Epochs → 85% | 2 Epochs → 94% | 5× Faster |

---

## 📁 Repository Structure

```text
cat-dog-classifier/
│
├── models/
│   ├── baseline_cnn.keras                  # Saved scratch-trained CNN checkpoint
│   ├── mobilenetv2_final.keras             # Saved fine-tuned MobileNetV2 checkpoint
│   └── confusion_matrix_comparison.png     # Evaluated dual confusion matrix plot
│
├── app.py                 # Streamlit UI with ImageNet synset semantic gatekeeper
├── download_data.py       # JFIF byte-level integrity verification & directory prep
├── train_scratch_cnn.py   # Baseline 3-block CNN training pipeline
├── train_mobilenet.py     # 2-phase progressive fine-tuning pipeline
├── evaluate_models.py     # Multi-class classification report & Seaborn heatmap generator
├── requirements.txt       # Dependencies (TensorFlow, Streamlit, Scikit-learn, Seaborn)
├── .gitignore             # Excluded datasets, virtual environments, and caches
└── README.md              # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository & Set Up Virtual Environment

```bash
git clone [https://github.com/soumilichanda/cat-dog-classifier.git](https://github.com/soumilichanda/cat-dog-classifier.git)
cd cat-dog-classifier

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation & Byte Sanitization

Extract the Microsoft Cats vs Dogs archive into `datasets/`, then run the header verification pipeline:

```bash
python download_data.py
```

### 3. Model Training

```bash
# Train baseline scratch CNN (10 epochs)
python train_scratch_cnn.py

# Train MobileNetV2 with two-phase fine-tuning (6 + 4 epochs)
python train_mobilenet.py
```

### 4. Evaluate Models

Generate classification reports and comparative confusion matrices:

```bash
python evaluate_models.py
```

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

---

## 🛡️ Out-of-Distribution (OOD) Gatekeeping Logic

Binary sigmoid classifiers output high probabilities ($>80\%$) even on non-pet inputs (e.g., smartphones, fruit). This system implements a pre-inference verification stage:

$$\mathcal{I}_{\text{pet}} = \{151, \dots, 268\} \cup \{281, \dots, 285\}$$

If an image’s top ImageNet predictions do not contain a domestic canine or feline synset with $P \ge 0.08$, execution halts and routes to the **"Else / Other"** warning state, preventing false positive animal classifications on arbitrary objects.

---

## 👤 Author

- **Soumili Chanda** 
