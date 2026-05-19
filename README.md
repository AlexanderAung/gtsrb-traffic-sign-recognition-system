# GTSRB Traffic Sign Recognition

## Setup

This project is intentionally small so a beginner can read the full pipeline:

```bash
uv sync
```

Required dependencies are kept minimal:

- `torch`
- `torchvision`
- `pillow`
- `numpy`

The dataset root should already exist at:

```text
traffic_signs/
```

For training, the code uses:

```text
traffic_signs/Train/<class_id>/<image>.png
```

That folder layout works directly with `torchvision.datasets.ImageFolder`.

## Project Structure

```text
gtsrb_project/
├── dataset.py
├── model.py
├── train.py
├── predict.py
└── utils.py
traffic_signs/
├── Train/
├── Test/
├── Train.csv
└── Meta.csv
```

## Train

Run training from the project root:

```bash
uv run python gtsrb_project/train.py
```

Expected behavior:

- Loads images from `traffic_signs/Train`
- Trains a small CNN for 5 epochs
- Prints loss and accuracy once per epoch
- Saves the trained model to `traffic_sign_model.pth`

Example output:

```text
Epoch 1/5 - Loss: 2.1234 - Accuracy: 0.4123
Epoch 2/5 - Loss: 1.2345 - Accuracy: 0.6500
Training finished. Model saved to traffic_sign_model.pth
```

## Predict

After training, run prediction on one image:

```bash
uv run python gtsrb_project/predict.py traffic_signs/Test/00000.png
```

Expected behavior:

- Loads `traffic_sign_model.pth`
- Preprocesses the image to `32x32`
- Prints the predicted class folder name

Example output:

```text
Predicted class: 25
```
