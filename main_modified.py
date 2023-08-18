import torch
from torchvision import models
import torch.nn as nn
from filefunctions_modified import buildDataset, STFT_CustomDataset
from ModelFuctions import train, test
from sklearn.model_selection import train_test_split
from custom_net_modified import CustomNet

# Hyperparameters
batch_size = 100
epoch_num = 40
learning_rate = 1e-03

# MODEL AND OPTIMISER
# net = models.alexnet()
# net.classifier[6] = nn.Linear(4096, 1)  # Output layer for regression
# net = CustomNet(signal_shape, )
# net = models.resnet50() # You can set pretrained=True if you want to use pretrained weights
# net.fc = nn.Linear(net.fc.in_features, 1) # Output layer for regression


act = "relu"
reg = 0
Filters = [48,96,192]
FilterSize = [3,3]
DenseLayers = [48]
fc_drop = 0.3
conv_drop = 0



# net = CustomNet(signal_shape, Filters, FilterSize, DenseLayers, fc_drop, act)


criterion = nn.MSELoss()               # Mean squared error for regression
# optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def main():
    signal_file = "Data/Sim_GW_Signals.h5"
    label_file = "Data/Sim_GW_Lables.h5"
    Arr, GT_normalized, norms, _ = buildDataset(signal_file, label_file)  # Ignore the signal_shape returned

    # Splitting data
    DAT_train, DAT_test, GT_train, GT_test = train_test_split(Arr, GT_normalized, test_size=0.20)

    # Create STFT_CustomDataset instances
    train_data = STFT_CustomDataset(DAT_train, GT_train)
    test_data = STFT_CustomDataset(DAT_test, GT_test)

    # Get the processed data shape from a sample in the dataset
    sample_data, _ = train_data[0]
    input_shape = sample_data.shape
    print(input_shape)

    # # Initialize CustomNet using the correct input shape
    # net = CustomNet(input_shape, Filters, FilterSize, DenseLayers, fc_drop, act)
    # # Initialize the existing nets
    # net = models.alexnet()

    net = models.resnet50(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 1)  # For regression
    print(net.fc.in_features)
    # Freeze all layers except the last few layers for fine-tuning
    #for param in net.parameters():
        #param.requires_grad = False

    # net.classifier[6] = nn.Linear(4096, 1)  # Output layer for regression
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)



    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=1)

    # Training and testing
    train(net, train_loader, epoch_num, optimizer, criterion, batch_size)
    test(test_loader, batch_size, norms) # Pass norms to test function

if __name__ == '__main__':
    main()
