"""
utils/preprocessing.py
======================
Per-model image preprocessing utilities.

IMPORTANT — Each Keras application family has its own `preprocess_input`
function that expects a specific pixel-value range.  Using the wrong one
is a common source of silent accuracy degradation.

  ┌─────────────────┬────────────────────────────────────────────────────┐
  │ Model           │ preprocess_input behaviour                         │
  ├─────────────────┼────────────────────────────────────────────────────┤
  │ InceptionV3     │ Scales [0, 255] → [−1, 1]  (subtract 127.5, /127.5)│
  │ ConvNeXt-Base   │ Near pass-through — model was saved with           │
  │                 │ include_preprocessing=True, so the normalisation    │
  │                 │ layer is already baked in.  We still call the       │
  │                 │ official helper so the dtype is correct.            │
  │ EfficientNet    │ Scales [0, 255] → [0, 1]  (divides by 255)         │
  └─────────────────┴────────────────────────────────────────────────────┘
"""

import numpy as np
from PIL import Image


def preprocess_for_inceptionv3(image: Image.Image, target_size=(299, 299)) -> np.ndarray:
    """
    Prepare a PIL image for InceptionV3 inference.

    Steps:
      1. Resize to 299×299 (InceptionV3's required input).
      2. Convert to float32 array shaped (299, 299, 3).
      3. Add batch dimension → (1, 299, 299, 3).
      4. Apply InceptionV3's preprocess_input: pixels scaled to [−1, 1].

    Args:
        image: RGB PIL Image of any size.
        target_size: (H, W) tuple; defaults to (299, 299).

    Returns:
        numpy array of shape (1, H, W, 3), dtype float32, values in [−1, 1].
    """
    from tensorflow.keras.applications.inception_v3 import preprocess_input  # noqa: E402

    img = image.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # (H, W, 3)
    arr = np.expand_dims(arr, axis=0)              # (1, H, W, 3)
    arr = preprocess_input(arr)                    # scales to [-1, 1]
    return arr


def preprocess_for_convnext(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Prepare a PIL image for ConvNeXt-Base inference.

    The model was saved with `include_preprocessing=True`, meaning the
    internal normalisation layers are already part of the saved graph.
    Calling `convnext.preprocess_input` is still correct — it does a
    minimal dtype cast without destructive rescaling when the baked-in
    layers handle the rest.

    Steps:
      1. Resize to 224×224.
      2. Convert to float32 array shaped (224, 224, 3).
      3. Add batch dimension → (1, 224, 224, 3).
      4. Apply ConvNeXt's preprocess_input (near pass-through for 0–255 float).

    Args:
        image: RGB PIL Image of any size.
        target_size: (H, W) tuple; defaults to (224, 224).

    Returns:
        numpy array of shape (1, H, W, 3), dtype float32.
    """
    from tensorflow.keras.applications.convnext import preprocess_input  # noqa: E402

    img = image.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def preprocess_for_efficientnet(image: Image.Image, target_size: tuple) -> np.ndarray:
    """
    Prepare a PIL image for EfficientNet inference.

    Input size is deliberately NOT hardcoded here because several EfficientNet
    variants exist (B0: 224, B1: 240, B2: 260, B3: 300 …).  The caller is
    expected to pass the exact `target_size` detected from `model.input_shape`
    at load time.

    Steps:
      1. Resize to `target_size`.
      2. Convert to float32 array.
      3. Add batch dimension.
      4. Apply EfficientNet's preprocess_input: scales [0, 255] → [0, 1].

    Args:
        image: RGB PIL Image of any size.
        target_size: (H, W) tuple read from model.input_shape.

    Returns:
        numpy array of shape (1, H, W, 3), dtype float32, values in [0, 1].
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input  # noqa: E402

    img = image.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr
