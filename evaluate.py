# evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SwinTransformerModel
from inference import inference
from evaluation_utils import calculate_accuracy, plot_confusion_matrix

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())

    accuracy = calculate_accuracy(true_labels, predicted_labels)
    print(f"Accuracy: {accuracy:.4f}")

    plot_confusion_matrix(true_labels, predicted_labels, class_names)

if __name__ == "__main__":
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    num_classes = 10
    batch_size = 64

    # Initialize the dataset and dataloaders
    val_dataset = CustomDataset(data_dir='path/to/val_data', transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SwinTransformerModel(num_classes=num_classes).to(device)

    # Load the trained model
    checkpoint = torch.load('trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate the model
    evaluate_model(model, val_loader, criterion, device)
