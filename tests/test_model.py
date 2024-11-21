import torch
import pytest
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import MNIST_1, count_parameters

def test_parameter_count():
    """
    Test that the model has fewer than 25,000 parameters.
    """
    model = MNIST_1()
    assert count_parameters(model) < 25000, "Model has more than 25000 parameters"

def test_input_size():
    """
    Test that the model produces the correct output shape for a 28x28 input.
    """
    model = MNIST_1()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output shape is incorrect"

def test_batch_input():
    """
    Test that the model produces the correct output shape for a batch of inputs.
    """
    model = MNIST_1()
    input_tensor = torch.randn(32, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (32, 10), "Output shape for batch input is incorrect"

def test_gradient_flow():
    """
    Test that the model produces the correct output shape for a batch of inputs.
    """
    model = MNIST_1()
    input_tensor = torch.randn(1, 1, 28, 28, requires_grad=True)
    output = model(input_tensor)
    output[0, 0].backward()
    assert input_tensor.grad is not None, "Gradients are not flowing through the model"

def test_output_range():
    """
    Test that the model produces probabilities in the range [0, 1].
    """
    model = MNIST_1()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    
    # Convert log probabilities to probabilities
    probabilities = torch.exp(output)
    
    # Check if probabilities are in the range [0, 1]
    assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1), "Model output is not in the range [0, 1]"
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5), "Model output does not sum to 1"

def test_noisy_input():
    """
    Test that the model produces the correct output shape for a noisy input.
    """
    model = MNIST_1()
    input_tensor = torch.randn(1, 1, 28, 28) + torch.randn(1, 1, 28, 28) * 0.1
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output shape with noisy input is incorrect"

def test_model_accuracy():
    """
    Test that the model achieves at least 95% accuracy on the test set.
    """
    # Set up the data loader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Initialize the model
    model = MNIST_1()

    # Load the best saved model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / total
    assert accuracy >= 95, f"Accuracy did not reach 95% with the best saved model, got {accuracy:.2f}%"
