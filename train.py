# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CustomDataset
from model import SwinTransformerModel

def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs):
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss:.4f}")

        # Validation loop
        model.eval()
        total_correct = 0
        total_samples = 0
        valid_loss = 0.0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        average_valid_loss = valid_loss / len(valid_loader)
        accuracy = total_correct / total_samples

        print(f"Epoch [{epoch+1}/{num_epochs}], Valid Loss: {average_valid_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained model and other necessary information
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, 'trained_model.pth')

if __name__ == "__main__":
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Define hyperparameters
    num_classes = 10
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10

    # Initialize the dataset and dataloaders
    full_dataset = CustomDataset(data_dir='train', transform=transforms.ToTensor())

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = SwinTransformerModel(num_classes=num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model and validate
    train_model(model, train_loader, valid_loader, criterion, optimizer, "cuda", num_epochs)
