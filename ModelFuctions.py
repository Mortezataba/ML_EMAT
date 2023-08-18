import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train(net, train_loader, epoch_num, optimizer, criterion, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')  # Print the name of the GPU being used
    else:
        print('Using CPU')
    net.to(device)
    criterion = criterion.to(device)

    net.train()

    loss_list = []  # List to store loss values for each epoch

    for epoch in range(epoch_num):
        running_loss = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)

                # Add this line to reshape the labels
                labels = labels.view(-1, 1)  # Reshape to (batch_size, 1)
                # Run the forward pass
                outputs = net(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Backprop and perform Adam optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the loss
                tepoch.set_postfix(loss=loss.item())

        loss_list.append(running_loss / len(train_loader))  # Average loss for the epoch

    # Plot the loss curve
    plt.plot(loss_list)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.show()

    modsav = 'model' + '.pt'
    torch.save(net, modsav)


def test(test_loader, batch_size, norms):
    print('Testing')
    model_n = 'model.pt'
    net = torch.load(model_n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    mse_loss = 0.0
    criterion = torch.nn.MSELoss()  # Mean squared error for evaluation
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1) # Labels as column vector
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            mse_loss += loss.item() * inputs.size(0)
            predictions.extend(outputs.cpu().numpy().flatten())
            ground_truths.extend(labels.cpu().numpy().flatten())
        mse_loss /= len(test_loader.dataset)
        print(f'Mean Squared Error on Test Data: {mse_loss}')

    # Unnormalize the predictions and ground truths
    predictions = [norms['YNorm'] * pred + norms['YShift'] for pred in predictions]
    ground_truths = [norms['YNorm'] * gt + norms['YShift'] for gt in ground_truths]

    # Plotting Predicted vs Ground Truth Labels
    plt.figure(figsize=(10, 5))
    plt.scatter(ground_truths, predictions, alpha=0.5)
    plt.plot([min(ground_truths), max(ground_truths)], [min(ground_truths), max(ground_truths)], color='red')
    plt.xlabel('Ground Truth Labels')
    plt.ylabel('Predicted Labels')
    plt.title('Predicted vs Ground Truth Labels')
    plt.show()

    # Plotting Histogram of Prediction Errors
    errors = [pred - gt for pred, gt in zip(predictions, ground_truths)]
    plt.figure(figsize=(10, 5))
    plt.hist(errors, bins=20, alpha=0.5)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Errors')
    plt.show()
