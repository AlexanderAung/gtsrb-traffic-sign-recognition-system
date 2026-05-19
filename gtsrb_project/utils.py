# Senior Engineer Note:
# Utility functions for training and evaluation.


# This function measures how many predictions match the true labels in one batch.
def calculate_accuracy(outputs, labels):
    predicted_labels = outputs.argmax(dim=1)
    correct_predictions = (predicted_labels == labels).sum().item()
    total_predictions = labels.size(0)

    return correct_predictions / total_predictions
