# Simple Neural Network for MNIST

This repository contains a simple neural network implemented in PyTorch to classify MNIST digits and a CI/CD pipeline to train the model and push it to the repository.

## Requirements

- Python 3.8
- PyTorch
- torchvision
- pytest
- torchinfo

## Usage

1. Install the required packages.
2. Run `python src/train.py` to train the model.
3. Run `pytest` to execute tests.

## Tests

The test suite includes various tests to ensure the model's functionality and robustness:

- **Parameter Count Test**: Verifies that the model has fewer than 25,000 parameters.
- **Input Size Test**: Ensures the model produces the correct output shape for a single input.
- **Batch Input Test**: Checks the model's ability to handle batch inputs.
- **Gradient Flow Test**: Confirms that gradients are flowing through the model during backpropagation.
- **Output Range Test**: Validates that the model's output is a valid probability distribution.
- **Noisy Input Test**: Tests the model's robustness to noisy inputs.
- **Accuracy Test**: Ensures that the model reaches at least 95% accuracy in the first epoch of training on the MNIST dataset.

### Explanation

The accuracy test is designed to verify that the model can achieve a high level of accuracy quickly. It sets up a data loader for the MNIST dataset, initializes the model, and trains it for one epoch. The test then calculates the accuracy and asserts that it reaches at least 95%. This test helps ensure that the model is effectively learning from the data and performing well.

Note: The accuracy test can be computationally intensive and may vary in results due to factors like random initialization and learning rate. It is recommended to run this test in an environment with sufficient resources.