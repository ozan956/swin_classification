# inference.py
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import SwinTransformerModel

def inference(model, data_loader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    return predictions

if __name__ == "__main__":
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define hyperparameters
    num_classes = 10
    batch_size = 64

    # Initialize the dataset and dataloaders
    inference_dataset = CustomDataset(data_dir='path/to/inference_data', transform=transforms.ToTensor())
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SwinTransformerModel(num_classes=num_classes).to(device)

    # Load the trained model
    checkpoint = torch.load('trained_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Perform inference
    inference(model, inference_loader, device)
