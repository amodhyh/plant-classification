# Plant Leaf Classification using Deep Learning

This repository contains the codebase for fine-tuning deep learning models on the Egyptian Plant Leaf Image Dataset (EgyPLI). This branch (`yasitha`) specifically focuses on training and evaluating an **EfficientNet-B2** architecture.

## Overview
The goal of this project is to accurately classify plant leaf images into their respective species categories. To achieve robust performance, we employ a deep transfer learning methodology leveraging the EfficientNet-B2 backbone.

### Key Features Implemented:
* **Model Architecture**: 
  * Pre-trained **EfficientNet-B2** backbone.
  * Global Average Pooling (GAP) applied to the final hidden states.
  * Multi-Sample Dropout (5 parallel paths with `p=0.5`) converging into a shared Output Dense Layer to improve generalization and stability.
* **Training Methodology**:
  * **Gradual Unfreezing**: Initial training focuses purely on the classifier head with the backbone frozen. The backbone is then unfrozen halfway through training with a reduced learning rate.
  * **Learning Rate Scheduler**: AdamW optimizer paired with Cosine Annealing with Warmup.
* **Data Pipeline & Augmentation**: 
  * Heavy data augmentation applied strictly to the training split: random rotations (25 degrees), affine transformations (translation, scaling, shearing), horizontal flips, and brightness jittering.
  * No augmentations are applied to validation/test sets to maintain evaluation integrity.
* **Evaluation Strategy**:
  * **Holdout Test Set**: A strictly untouched 10% slice of the original dataset is held out for final evaluation.
  * **Stratified K-Fold Cross Validation**: The remaining 90% is split using 5-Fold Stratified CV to ensure uniform class distribution across folds. The best model weights per fold are tracked and restored.

## Getting Started

### Prerequisites
Make sure to install the required libraries (the project uses PyTorch):
```bash
pip install torch torchvision transformers scikit-learn nbformat
```

### Dataset Structure
The notebook expects the data to be placed in the `data/` directory categorized by class sub-directories:
```
data/
├── Apple/
├── Berry/
├── Fig/
...
```

### Running the Notebook
Open and run `efficientNet.ipynb` to execute the full K-Fold training pipeline and test evaluation sequentially.