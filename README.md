# Simple Neural Network for MNIST

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-3810/)  [![Python application](https://github.com/KD1994/session-5-MNIST-CICD/actions/workflows/python-app.yml/badge.svg)](https://github.com/KD1994/session-5-MNIST-CICD/actions/workflows/python-app.yml)  ![Build Status](https://img.shields.io/badge/build-pass-green)


This repository contains a simple neural network implemented in PyTorch to classify MNIST digits and a CI/CD pipeline to train the model and push it to the repository.

## Requirements

- Python 3.8
- PyTorch
- torchvision
- pytest
- torchinfo

## Usage

1. Install the required packages.
2. Run `python train.py` to train the model.
3. Run `pytest -v` to execute tests.


## Network Architecture

| INPUT    | KERNEL     | OUTPUT |
|----------|------------|--------|
| 28x28x1  | 3x3x1x8    | 26     |
| 26x26x8  | 3x3x8x16   | 24     |
| Maxpool()|            | 12     |
| 24x24x16 | 3x3x16x32  | 10     |
| Maxpool()|            | 5      |
| 5x5x32   | 3x3x32x32  | 3      |
| 3x3x32   | 2x2x32x32  | 2      |
| Flatten()|            |        | 


## Model Summary

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
MnistNet                                 [1, 10]                   --
├─Sequential: 1-1                        [1, 128]                  --
│    └─Conv2d: 2-1                       [1, 8, 26, 26]            80
│    └─GELU: 2-2                         [1, 8, 26, 26]            --
│    └─BatchNorm2d: 2-3                  [1, 8, 26, 26]            16
│    └─Conv2d: 2-4                       [1, 16, 24, 24]           1,168
│    └─GELU: 2-5                         [1, 16, 24, 24]           --
│    └─BatchNorm2d: 2-6                  [1, 16, 24, 24]           32
│    └─MaxPool2d: 2-7                    [1, 16, 12, 12]           --
│    └─GELU: 2-8                         [1, 16, 12, 12]           --
│    └─BatchNorm2d: 2-9                  [1, 16, 12, 12]           32
│    └─Conv2d: 2-10                      [1, 32, 10, 10]           4,640
│    └─GELU: 2-11                        [1, 32, 10, 10]           --
│    └─BatchNorm2d: 2-12                 [1, 32, 10, 10]           64
│    └─MaxPool2d: 2-13                   [1, 32, 5, 5]             --
│    └─GELU: 2-14                        [1, 32, 5, 5]             --
│    └─BatchNorm2d: 2-15                 [1, 32, 5, 5]             64
│    └─Conv2d: 2-16                      [1, 32, 3, 3]             9,248
│    └─GELU: 2-17                        [1, 32, 3, 3]             --
│    └─BatchNorm2d: 2-18                 [1, 32, 3, 3]             64
│    └─Conv2d: 2-19                      [1, 32, 2, 2]             4,128
│    └─Dropout2d: 2-20                   [1, 32, 2, 2]             --
│    └─Flatten: 2-21                     [1, 128]                  --
├─Sequential: 1-2                        [1, 10]                   --
│    └─Linear: 2-22                      [1, 10]                   1,290
==========================================================================================
Total params: 20,826
Trainable params: 20,826
Non-trainable params: 0
Total mult-adds (M): 1.29
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.32
Params size (MB): 0.08
Estimated Total Size (MB): 0.40
==========================================================================================
```

## Training

```
Batch: 0, Loss: 2.4165, Accuracy: 3.12%
Batch: 100, Loss: 0.3989, Accuracy: 78.96%
Batch: 200, Loss: 0.2644, Accuracy: 84.95%
Batch: 300, Loss: 0.3112, Accuracy: 87.67%
Batch: 400, Loss: 0.2250, Accuracy: 89.20%
Batch: 500, Loss: 0.1895, Accuracy: 90.39%
Batch: 600, Loss: 0.0442, Accuracy: 91.12%
Batch: 700, Loss: 0.3401, Accuracy: 91.71%
Batch: 800, Loss: 0.1248, Accuracy: 92.18%
Batch: 900, Loss: 0.1686, Accuracy: 92.61%
```


## Tests

The test suite includes various tests to ensure the model's functionality and robustness:

- **Parameter Count Test**: Verifies that the model has fewer than 25,000 parameters.
- **Input Size Test**: Ensures the model produces the correct output shape for a single input.
- **Batch Input Test**: Checks the model's ability to handle batch inputs.
- **Gradient Flow Test**: Confirms that gradients are flowing through the model during backpropagation.
- **Output Range Test**: Validates that the model's output is a valid probability distribution.
- **Noisy Input Test**: Tests the model's robustness to noisy inputs.
- **Accuracy Test**: Ensures that the model reaches at least 95% accuracy in the first epoch of training on the MNIST dataset.

<<<<<<< HEAD
### Explanation

The accuracy test is designed to verify that the model can achieve a high level of accuracy quickly. It sets up a data loader for the MNIST dataset, initializes the model, and trains it for one epoch. The test then calculates the accuracy and asserts that it reaches at least 95%. This test helps ensure that the model is effectively learning from the data and performing well.

Note: The accuracy test can be computationally intensive and may vary in results due to factors like random initialization and learning rate. It is recommended to run this test in an environment with sufficient resources.
=======
>>>>>>> train-tweak
