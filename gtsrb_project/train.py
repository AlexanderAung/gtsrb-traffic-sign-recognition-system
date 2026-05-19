# Senior Engineer Note:
# This module handles the training loop for GTSRB classification.

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloader
from model import TrafficSignCNN
from utils import calculate_accuracy


# This function saves the trained weights and class names needed for later prediction.
def save_checkpoint(model, class_names, path="traffic_sign_model.pth"):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }
    torch.save(checkpoint, path)


# This function runs the full training pipeline from data loading to saved model file.
def train(dataset_root="traffic_signs", epochs=5, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = create_dataloader(dataset_root, batch_size=batch_size, shuffle=True)
    class_names = train_loader.dataset.classes

    model = TrafficSignCNN(num_classes=len(class_names)).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += calculate_accuracy(outputs, labels)

        average_loss = running_loss / len(train_loader)
        average_accuracy = running_accuracy / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {average_loss:.4f} "
            f"- Accuracy: {average_accuracy:.4f}"
        )

    save_checkpoint(model, class_names)
    print("Training finished. Model saved to traffic_sign_model.pth")


# This function is the terminal entry point that starts training with default settings.
def main():
    train()


if __name__ == "__main__":
    main()
