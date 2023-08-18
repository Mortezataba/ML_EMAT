import torch.nn as nn
import torch.nn.functional as F
import torch


class CustomNet(nn.Module):
    def __init__(self, signal_shape, Filters, FilterSize, DenseLayers, fc_drop, act='relu'):
        super(CustomNet, self).__init__()
        self.convs = nn.ModuleList()
        for ii in range(len(Filters)):
            self.convs.append(nn.Conv2d(in_channels=Filters[ii - 1] if ii != 0 else 3, out_channels=Filters[ii],
                                        kernel_size=FilterSize, stride=1, padding="same"))

            self.convs.append(
                nn.Conv2d(in_channels=Filters[ii], out_channels=Filters[ii], kernel_size=FilterSize, stride=1,
                          padding="same"))
            self.convs.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Dynamically compute the flattened size
        print(signal_shape)
        dummy_input = torch.randn(1, 3, *signal_shape[1:])

        dummy_output = self.forward_conv(dummy_input)
        self.flat_size = dummy_output.view(-1).shape[0]

        self.denses = nn.ModuleList()
        for ii in range(len(DenseLayers)):
            in_size = self.flat_size if ii == 0 else DenseLayers[ii - 1]
            self.denses.append(nn.Dropout(p=fc_drop))
            self.denses.append(nn.Linear(in_size, DenseLayers[ii]))
            if act == 'relu':
                self.denses.append(nn.ReLU())

        self.final_drop = nn.Dropout(p=fc_drop)
        self.output_layer = nn.Linear(DenseLayers[-1], 1)  # Output layer for regression

    # Function to only forward through convolutional layers
    def forward_conv(self, x):
        for layer in self.convs:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def forward(self, x):
        x = self.forward_conv(x)
        flattened_size = x.shape[1] * x.shape[2] * x.shape[3]

        x = x.view(-1, flattened_size)
        for layer in self.denses:
            x = layer(x)
        x = self.final_drop(x)
        x = self.output_layer(x)
        return x
