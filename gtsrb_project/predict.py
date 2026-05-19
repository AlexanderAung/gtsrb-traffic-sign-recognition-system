# Senior Engineer Note:
# This module runs inference on a single traffic sign image.

import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import TrafficSignCNN


# This function loads the saved model weights and class names from training.
def load_trained_model(model_path="traffic_sign_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]

    model = TrafficSignCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names, device


# This function prepares one image so it has the same tensor shape used in training.
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    image_tensor = transform(image)

    return image_tensor.unsqueeze(0)


# This function runs the model on one image and returns the predicted class name.
def predict(image_path, model_path="traffic_sign_model.pth"):
    model, class_names, device = load_trained_model(model_path)
    image_tensor = load_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_index = outputs.argmax(dim=1).item()

    predicted_class = class_names[predicted_index]
    print(f"Predicted class: {predicted_class}")

    return predicted_class


# This function reads the image path from the terminal in a beginner-friendly way.
def parse_args():
    parser = argparse.ArgumentParser(description="Predict one GTSRB traffic sign image.")
    parser.add_argument("image_path", help="Path to one image, for example traffic_signs/Test/00000.png")
    parser.add_argument("--model-path", default="traffic_sign_model.pth", help="Path to the trained model file.")

    return parser.parse_args()


# This function is the terminal entry point that reads arguments and starts prediction.
def main():
    args = parse_args()
    predict(args.image_path, model_path=args.model_path)


if __name__ == "__main__":
    main()
