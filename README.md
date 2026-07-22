# 🌿 Plant Leaf Classifier — 3-Model Comparison

A production-style Flask web app that classifies plant leaf images using **three independently trained Keras models** (InceptionV3, ConvNeXt-Base, EfficientNet) and displays their predictions side-by-side.

**Dataset:** EgyPLI (Egyptian Plant Leaf Images) · 8 classes: Apple, Berry, Fig, Guava, Orange, Palm, Persimmon, Tomato

---

## Project Structure

```
plant leaf class/
├── app.py                          # Flask app — routes, config, startup
├── requirements.txt
├── README.md
├── models/                         # ← Drop your .keras files here
│   ├── inceptionv3_egypli_final.keras
│   ├── convnext_base_egypli_final.keras
│   └── efficientnet_egypli_final.keras
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Per-model resize + preprocess_input
│   └── inference.py                # Model loading + prediction logic
├── templates/
│   └── index.html                  # Single-page UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## Quick Start

### 1. Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow can be large (~600 MB). Install time may vary.
> If you have a GPU and want CUDA acceleration, install `tensorflow[and-cuda]` instead.

### 3. Place your model files

Copy your three trained `.keras` (or `.h5`) model files into the `models/` folder:

| File name expected | Model |
|---|---|
| `inceptionv3_egypli_final.keras` | InceptionV3 |
| `convnext_base_egypli_final.keras` | ConvNeXt-Base |
| `efficientnet_egypli_final.keras` | EfficientNet |

> If a model file is missing the app will log a warning and continue — results for that model will show an error card in the UI.

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## How It Works

| Model | Input size | Preprocessing |
|---|---|---|
| InceptionV3 | 299×299 | Scales pixels to [−1, 1] |
| ConvNeXt-Base | 224×224 | Near pass-through (preprocessing baked in) |
| EfficientNet | Auto-detected from `model.input_shape` | Scales pixels to [0, 1] |

- All models are loaded **once at startup** into memory.
- Uploaded images are processed **entirely in memory** (no disk writes).
- Results are returned as JSON and rendered client-side with animated probability bars.
- If two or more models predict different classes, a **"Models disagree"** badge appears.

---

## Configuration

Edit the top of `app.py` to change:

- `CLASS_NAMES` — class label list (must match training order)
- `MAX_UPLOAD_BYTES` — max file size (default 10 MB)
- `MODEL_CONFIGS` — model file paths and input sizes
- `ALLOWED_EXTENSIONS` — accepted file types

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Model file not found` | Ensure `.keras` files are in `models/` with exact names |
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` inside your venv |
| App starts but predictions are wrong | Double-check class ordering matches training |
| EfficientNet card shows error | Model file may be corrupted or incompatible — re-export from training notebook |
