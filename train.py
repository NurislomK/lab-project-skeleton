import torch
import torch.nn as nn
import torch.optim as optim

from dataset.loader import get_dataloaders
from models.model import SimpleCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")

    for epoch in range(2):  # keep small for testing
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()