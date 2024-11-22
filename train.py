import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchinfo import summary
from src.model import MnistNet

def train():
    """
    Train the MnistNet model on the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Save a batch of transformed images
    images, _ = next(iter(train_loader))
    save_image(images, 'transformed_batch.jpg')

    model = MnistNet()
    summary(model, input_size=(1, 1, 28, 28))
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    best_model_path = 'best_model.pth'

    model.train()
    total = 0
    correct = 0
    for b_id, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()

        if b_id % 100 == 0:
            print(
                f"Batch: {b_id}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%"
            )
    torch.save(model.state_dict(), best_model_path)


if __name__ == "__main__":
    train() 