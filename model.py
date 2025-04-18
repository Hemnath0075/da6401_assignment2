# model.py
import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, 
                 channels=3,
                 num_filters=64,
                 kernel_size=3,
                 activation_fn=nn.ReLU,
                 dense_neurons=256,
                 num_classes=10):
        super(CustomCNN, self).__init__()

        self.activation_fn = activation_fn

        layers = []
        current_channels = channels
        for _ in range(5):
            layers.append(nn.Conv2d(current_channels, num_filters, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(kernel_size=2))
            current_channels = num_filters

        self.conv_layers = nn.Sequential(*layers)
        self.flattened_size = self._get_flattened_size(channels)
        self.fc1 = nn.Linear(self.flattened_size, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)

    def _get_flattened_size(self, in_channels):
        dummy_input = torch.zeros(1, in_channels, 224, 224)
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        return output.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.activation_fn()(self.fc1(x))
        x = self.fc2(x)
        return x
