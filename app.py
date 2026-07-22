"""
Plant Leaf Classification - Flask Web App
=========================================
Three-model comparison: InceptionV3, ConvNeXt-Base, EfficientNet
Dataset: EgyPLI (Egyptian Plant Leaf Images) - 8 classes
"""

# ── Suppress TensorFlow / oneDNN / absl noise BEFORE any TF import ────────────
# These must be set before `import tensorflow` is executed anywhere.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # 0=DEBUG,1=INFO,2=WARNING,3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # silence oneDNN floating-point notice
os.environ["ABSL_MIN_LOG_LEVEL"]    = "2"   # suppress absl INFO messages
# ──────────────────────────────────────────────────────────────────────────────

import io
import logging
from flask import Flask, request, jsonify, render_template
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────

# Exact class names in alphabetical order (must match training order)
CLASS_NAMES = ["Apple", "Berry", "Fig", "Guava", "Orange", "Palm", "Persimmon", "Tomato"]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_CONFIGS = {
    "InceptionV3": {
        "path": os.path.join(MODELS_DIR, "inceptionv3_egypli_final.keras"),
        "input_size": (299, 299),  # fixed for InceptionV3
        "auto_detect_size": False,
    },
    "ConvNeXt-Base": {
        "path": os.path.join(MODELS_DIR, "convnext_base_egypli_final.keras"),
        "input_size": (224, 224),  # fixed for ConvNeXt-Base
        "auto_detect_size": False,
    },
    "EfficientNet": {
        "path": os.path.join(MODELS_DIR, "efficientnet_egypli_final.keras"),
        "input_size": None,        # auto-detected from model.input_shape at load time
        "auto_detect_size": True,
    },
}

# ─── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("plant_leaf_app")

# ─── Flask App Init ────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# ─── Model Loading ─────────────────────────────────────────────────────────────

from utils.inference import load_all_models, predict_all

models_registry = {}


def init_models():
    """Load all three Keras models once at startup and store in registry."""
    global models_registry
    models_registry = load_all_models(MODEL_CONFIGS, logger)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    """Return True if the filename has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main upload / results page."""
    return render_template("index.html", class_names=CLASS_NAMES)


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accept a multipart file upload, run inference on all loaded models,
    and return JSON results for each model.

    Response schema:
    {
      "success": true,
      "results": {
        "InceptionV3":   { "predicted_class": str, "confidence": float, "probabilities": {class: float} },
        "ConvNeXt-Base": { ... },
        "EfficientNet":  { ... }  // or { "error": "..." } if model failed to load
      },
      "models_agree": bool
    }
    """
    # ── Validate file presence ──
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}."
        }), 400

    # ── Read image into memory (no disk write) ──
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.error("Failed to open uploaded image: %s", exc)
        return jsonify({"success": False, "error": "Could not read image file. Please upload a valid JPG or PNG."}), 400

    logger.info("Received classification request — image size: %s", image.size)

    # ── Run predictions ──
    try:
        results = predict_all(models_registry, image, CLASS_NAMES, logger)
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        return jsonify({"success": False, "error": "An internal error occurred during inference."}), 500

    # ── Check agreement ──
    top_preds = [
        res["predicted_class"]
        for res in results.values()
        if "predicted_class" in res
    ]
    models_agree = len(set(top_preds)) == 1 if top_preds else True

    return jsonify({
        "success": True,
        "results": results,
        "models_agree": models_agree,
    })


# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    logger.error("500 Internal Server Error: %s", error)
    return jsonify({"success": False, "error": "Internal server error."}), 500


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_models()
    logger.info("Starting Flask dev server on http://127.0.0.1:5000")
    app.run(debug=False, host="127.0.0.1", port=5000)
