# PneumoScan AI
### Pneumonia Detection from Chest X-Ray Images

A production-ready AI application that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using YOLOv8 image classification, with AI-generated clinical summaries powered by Sentence Transformers.

---

## Demo

> Upload a chest X-ray → Get instant classification + confidence score + AI clinical report

![PneumoScan UI](assets/demo.png)

---

## How It Works

```
Chest X-Ray Image
       ↓
YOLOv8m-cls (fine-tuned)
       ↓
Prediction: NORMAL / PNEUMONIA + Confidence Score
       ↓
Sentence Transformer (all-MiniLM-L6-v2)
       ↓
Cosine Similarity → Best matching clinical report template
       ↓
AI-Generated Clinical Summary displayed in UI
```

---

## Project Structure

```
Chectdetect/
├── model/
│   └── model.pkl              # Trained YOLOv8 model
├── service/
│   ├── __init__.py
│   └── pneumonia_backend.py   # Model loading, inference, report generation
├── ui/
│   ├── __init__.py
│   └── Pnumonia.py            # Streamlit UI
├── assets/                    # Static assets
├── script/                    # Utility scripts
├── app.py                     # Entry point
├── requirements.txt
└── README.md
```

---

##  Tech Stack

| Layer | Technology |
|-------|-----------|
| **Model** | YOLOv8m-cls (Ultralytics) |
| **Transfer Learning** | ImageNet pretrained weights |
| **NLP / Report** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Similarity** | Cosine Similarity (semantic template selection) |
| **UI** | Streamlit |
| **Training Platform** | Google Colab (T4 GPU) |
| **Language** | Python 3.12 |

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/pneumoscan-ai.git
cd pneumoscan-ai
```

### 2. Create and activate virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run ui/Pnumonia.py
```

---

## Requirements

```txt
streamlit
ultralytics
sentence-transformers
torch
torchvision
Pillow
numpy
```

Or install all at once:
```bash
pip install streamlit ultralytics sentence-transformers torch torchvision Pillow numpy
```

---

## Dataset

| Property | Details |
|----------|---------|
| **Source** | [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Classes** | NORMAL, PNEUMONIA |
| **Train set** | ~5,200 images |
| **Test set** | ~624 images |
| **Format** | JPEG grayscale |

### Dataset Split Used for Training

| Split | NORMAL | PNEUMONIA |
|-------|--------|-----------|
| Train | ~4,185 | ~3,550 |
| Val (carved from train) | ~740 | ~626 |
| Test | 234 | 390 |

> A 15% validation split was created from the training data since the original dataset contained no `val/` folder.

---

## Model Training

Training was performed on **Google Colab** with a **T4 GPU**.

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8m-cls.pt` |
| Epochs | 25 |
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Early stopping | patience = 5 |
| Augmentation | Enabled |

**Expected accuracy:** ~93–96% Top-1 on test set.

---

## AI Report Generation

After classification, the app uses **Sentence Transformers** to generate a contextual clinical summary:

1. A query is constructed: `"chest xray {label} diagnosis confidence {score}%"`
2. The query is encoded into a vector embedding
3. Cosine similarity is computed against a set of clinical report templates
4. The most semantically relevant template is selected and displayed

This approach ensures the output report is dynamically matched to the prediction context rather than being a static hardcoded string.

> Reports are AI-generated summaries for research purposes only. They do not constitute a medical diagnosis.

---

## Key Files

| File | Purpose |
|------|---------|
| `service/pneumonia_backend.py` | All logic: model loading, inference, report generation, templates |
| `ui/Pnumonia.py` | Streamlit UI: layout, components, styling |
| `model/model.pkl` | Trained YOLOv8 model weights |

---

## Disclaimer

This application is developed for **research and educational purposes only**.
It is not a certified medical device and must not be used for clinical diagnosis.
Always consult a licensed radiologist or physician for medical decisions.

---

## Author

**Yasiru**
PythonProject · Chectdetect · 2026