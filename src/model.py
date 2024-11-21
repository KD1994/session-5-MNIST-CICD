import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class MNIST_1(nn.Module):
    def __init__(self):
        """
        Initialize the MNIST_1 model.
        Calculate Output Shape: (Height + 2*Padding - dilation*(Kernel_size - 1) - 1) / Stride + 1
        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            # 28 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 26
            nn.GELU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            # 26 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 24

            nn.GELU(),
            nn.BatchNorm2d(16),
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            # 24 + 2*0 - 1*(2 - 1) - 1 / 2 + 1 = 12
            nn.MaxPool2d(2),
            # 24 + 2*0 - 1*(2 - 1) - 1 / 2 + 1 = 12

            nn.GELU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            # 12 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 10
            
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # 10 + 2*0 - 1*(2 - 1) - 1 / 2 + 1 = 5

            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            # 5 + 2*0 - 1*(3 - 1) - 1 / 1 + 1 = 3

            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2),
            # 3 + 2*0 - 1*(2 - 1) - 1 / 1 + 1 = 2

            nn.Dropout2d(0.4),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(32*2*2, 10)
        )
    
    def forward(self, x):
        """
        Forward pass of the MNIST_1 model.
        """
        x = self.conv1(x)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


def count_parameters(model):
    """
    Count the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    """
    Main function to count the number of parameters in the MNIST_1 model and print the summary.
    """
    model = MNIST_1()
    print(f"Number of parameters: {count_parameters(model)}") 

    summary(model, input_size=(1, 1, 28, 28))
