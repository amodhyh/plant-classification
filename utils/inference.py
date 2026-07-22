"""
utils/inference.py
==================
Model loading and prediction logic for all three plant-leaf classifiers.

Responsibilities:
  - Load each .keras model once at startup.
  - Auto-detect EfficientNet's input resolution from model.input_shape.
  - Route each image to the correct preprocessing function.
  - Return structured per-class probability dicts ready to be JSON-serialised.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from utils.preprocessing import (
    preprocess_for_inceptionv3,
    preprocess_for_convnext,
    preprocess_for_efficientnet,
)

logger = logging.getLogger("plant_leaf_app.inference")


# ─── Model Loading ─────────────────────────────────────────────────────────────

def _load_keras_model(path: str, model_name: str, log: logging.Logger):
    """
    Load a single Keras model from `path`.
    Returns the model object, or None on failure.
    """
    try:
        import tensorflow as tf  # deferred import so app starts even if TF missing
        model = tf.keras.models.load_model(path)
        log.info("✅ Loaded model '%s' from %s  |  input_shape=%s", model_name, path, model.input_shape)
        return model
    except FileNotFoundError:
        log.warning("⚠️  Model file not found for '%s': %s — model skipped.", model_name, path)
    except Exception as exc:
        log.error("❌ Failed to load model '%s': %s", model_name, exc)
    return None


def load_all_models(model_configs: dict, log: logging.Logger) -> dict:
    """
    Load all models defined in `model_configs` dict (from app.py).

    For EfficientNet (auto_detect_size=True), reads the spatial dimensions
    from `model.input_shape` after loading and stores them back into the
    registry entry so downstream preprocessing uses the correct size.

    Returns:
        registry dict:  { model_name: { "model": <Keras model|None>, "input_size": (H, W) } }
    """
    registry = {}

    for name, cfg in model_configs.items():
        model = _load_keras_model(cfg["path"], name, log)
        input_size = cfg["input_size"]

        if model is not None and cfg.get("auto_detect_size"):
            # input_shape is typically (None, H, W, C) or (None, C, H, W)
            shape = model.input_shape
            # Find the spatial dims — ignore None (batch) and 3 (channels)
            spatial = [d for d in shape if d not in (None, 3)]
            if len(spatial) >= 2:
                input_size = (int(spatial[0]), int(spatial[1]))
                log.info("EfficientNet input size auto-detected: %s", input_size)
            else:
                input_size = (224, 224)
                log.warning("Could not parse EfficientNet input_shape %s — defaulting to 224×224", shape)

        registry[name] = {"model": model, "input_size": input_size}

    loaded = sum(1 for v in registry.values() if v["model"] is not None)
    log.info("Models loaded: %d / %d", loaded, len(model_configs))
    return registry


# ─── Prediction ────────────────────────────────────────────────────────────────

def _softmax_to_dict(probs: np.ndarray, class_names: list) -> dict:
    """
    Convert a 1-D probability array to {class_name: rounded_float} dict,
    sorted descending by probability.
    """
    pairs = sorted(zip(class_names, probs.tolist()), key=lambda x: x[1], reverse=True)
    return {cls: round(prob, 6) for cls, prob in pairs}


def _predict_single(model, image: Image.Image, model_name: str, input_size: tuple, class_names: list) -> dict:
    """
    Run inference for one model.

    Dispatches to the correct preprocessing function based on model_name,
    then calls model.predict() and extracts the top class + full distribution.

    Args:
        model:       Loaded Keras model.
        image:       RGB PIL Image.
        model_name:  One of "InceptionV3", "ConvNeXt-Base", "EfficientNet".
        input_size:  (H, W) to resize the image to.
        class_names: Ordered list of class label strings.

    Returns:
        {
          "predicted_class": str,
          "confidence": float,          # top class probability as percentage
          "probabilities": {str: float} # all classes, sorted desc
        }
    """
    # ── Preprocessing (model-specific) ──────────────────────────────────────
    if model_name == "InceptionV3":
        arr = preprocess_for_inceptionv3(image, target_size=input_size)
    elif model_name == "ConvNeXt-Base":
        arr = preprocess_for_convnext(image, target_size=input_size)
    elif model_name == "EfficientNet":
        arr = preprocess_for_efficientnet(image, target_size=input_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # ── Inference ────────────────────────────────────────────────────────────
    preds = model.predict(arr, verbose=0)   # shape: (1, num_classes)
    probs = preds[0]                        # shape: (num_classes,)

    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    confidence = round(float(probs[top_idx]) * 100, 2)

    return {
        "predicted_class": top_class,
        "confidence": confidence,
        "probabilities": _softmax_to_dict(probs, class_names),
    }


def predict_all(
    registry: dict,
    image: Image.Image,
    class_names: list,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Run predictions on all models in the registry.

    Models that failed to load (model=None) return an error entry instead
    of a prediction, so the app still shows results from the other models.

    Args:
        registry:    Output of load_all_models().
        image:       RGB PIL Image uploaded by the user.
        class_names: Ordered class label list.
        log:         Logger instance.

    Returns:
        { model_name: {predicted_class, confidence, probabilities} | {error: str} }
    """
    results = {}

    for name, entry in registry.items():
        model = entry["model"]
        input_size = entry["input_size"]

        if model is None:
            results[name] = {"error": f"Model '{name}' is unavailable (failed to load at startup)."}
            continue

        try:
            result = _predict_single(model, image, name, input_size, class_names)
            log.info("Prediction — %s: %s (%.1f%%)", name, result["predicted_class"], result["confidence"])
            results[name] = result
        except Exception as exc:
            log.error("Prediction failed for model '%s': %s", name, exc)
            results[name] = {"error": f"Inference failed: {str(exc)}"}

    return results
