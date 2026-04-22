# Diabetic Retinopathy Detection

End-to-end deep learning pipeline for 5-class diabetic retinopathy (DR) severity grading from retinal fundus images, using EfficientNetV2-S with ordinal regression, model evaluation, Grad-CAM explainability, and a Streamlit app for interactive inference.

## Overview

This project predicts DR severity on an ordinal scale:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

Instead of plain multiclass softmax, the model uses ordinal regression with cumulative logits to better capture class ordering.

## Key Features

- EfficientNetV2-S backbone (via timm)
- Ordinal regression loss for ordered severity labels
- Mixed precision training support on CUDA
- QWK-based validation and checkpointing
- Validation metrics export (JSON) and confusion matrix plot
- Batch inference to CSV
- Grad-CAM heatmaps in a Streamlit interface

## Repository Structure

```text
dr_project/
	app_streamlit.py
	requirements.txt
	train_val_split.py
	clean_test_csv.py
	check_class_count.py
	src/
		dataset.py
		evaluate.py
		gradcam.py
		inference.py
		losses.py
		model.py
		train.py
		transforms.py
		utils.py
	data/
		train.csv
		train_split.csv
		val.csv
		test.csv
		test_clean.csv
		train_images/
		test_images/
	outputs/
		checkpoints/
```

## Environment Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Format

### Train/Validation CSV

Expected columns:

- image_id (string, without extension)
- label (integer in [0, 4])

The dataset loader reads images as:

```text
<images_dir>/<image_id>.png
```

### Test CSV

Expected column:

- image_id

## Data Preparation

Generate train/validation split from a Kaggle-style train CSV:

```bash
python train_val_split.py
```

This script maps:

- id_code -> image_id
- diagnosis -> label

Optional utilities:

```bash
python check_class_count.py
python clean_test_csv.py
```

## Training

Run training:

```bash
python src/train.py
```

Default behavior in the training script:

- Train CSV: data/train_split.csv
- Validation CSV: data/val.csv
- Image directory: data/train_images
- Checkpoint path: outputs/checkpoints/best.pth
- Epochs: 30
- Batch size: 8
- Optimizer: AdamW
- LR scheduler: CosineAnnealingLR
- Model selection metric: validation QWK

## Evaluation

Run evaluation on validation data:

```bash
python src/evaluate.py
```

Default outputs:

- outputs/eval_metrics_1.json
- outputs/confusion_matrix_1.png

Metrics include:

- QWK
- MAE
- RMSE
- Within-1 accuracy
- Accuracy
- Per-class classification report

## Inference

Run batch inference:

```bash
python src/inference.py
```

Default input/output in script:

- Model: outputs/checkpoints/best.pth
- Input CSV: data/test_clean.csv
- Image directory: data/test_images
- Predictions CSV: outputs/preds.csv

## Streamlit App

Launch the web app:

```bash
streamlit run app_streamlit.py
```

The app:

- Loads a trained checkpoint
- Accepts uploaded retinal images (png/jpg/jpeg)
- Predicts DR grade (0-4)
- Displays cumulative ordinal probabilities
- Generates and overlays Grad-CAM heatmaps

## Notes

- Ensure checkpoint path used in scripts exists before running inference or app.
- If image files are not .png, update the dataset extension handling in src/dataset.py.
- For CPU-only runs, scripts automatically fall back when CUDA is unavailable.
