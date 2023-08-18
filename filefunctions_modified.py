import pandas as pd
import numpy
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys
from PIL import Image
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def process_file(h1_data):

    # time window for different modes: SH0(7e-5 - 14.3e-5); SH1 (14.3e-5 - 2.206e-4)
    # or the whole useful part of the signal (7e-5 - 2.206e-4)
    t1 = 7e-5  # Start time in seconds,
    t2 = 14.3e-5  # End time in seconds
    f1 = 0  # Start frequency in Hz
    f2 = 350000  # End frequency in Hz

    # Define STFT parameters
    fs =  1 / 6.3999998e-08  # Sampling frequency
    nperseg = 128  # Window size
    noverlap = nperseg - 11  # Overlap between windows

    # Load MATLAB file
    #data = sio.loadmat(os.path.join(dir_path, filename))
    #h1_data = data['h1_data'][int(t1*fs):int(t2*fs), :]  # Select second column
    h1_data = h1_data[int(t1*fs):int(t2*fs)]
    # Perform STFT

    # f, t, Zxx = signal.spectrogram(h1_data, fs, nperseg=nperseg, noverlap=noverlap, nfft=4096,
    #                                detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')
    f, t, Zxx = signal.spectrogram(h1_data, fs, nperseg=nperseg, noverlap=noverlap, nfft=4096,
                                   detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='angle')

    # Apply frequency window
    f_idx = numpy.where((f >= f1) & (f <= f2))[0]
    data = Zxx[f_idx, :]


    # Custom normalization
    X_shift = np.mean(data[data > 0])
    X_norm = np.max(np.abs(data - X_shift))
    data_normalized = (data - X_shift) / X_norm
    #
    # # Convert to image
    # img = Image.fromarray(data_normalized)


    # Return the norms as well, if you need them later
    return data_normalized, {'XShift': X_shift, 'XNorm': X_norm}


########################################################################################################################
# Build Dataset
########################################################################################################################
''' Pull data from Image and GT files to build dataset - Designed for single and multiclass GT'''



def buildDataset(signal_file, label_file):
    print('Building Dataset')
    Arr = []  # List to store the processed images
    GT = []  # List to store the ground truth (defect_depth)

    # Reading the signal data from HDF5 file
    with h5py.File(signal_file, 'r') as f_signal:
        signal_data = np.array(f_signal['signals'])

    # Reading the label data from HDF5 file
    with h5py.File(label_file, 'r') as f_labels:
        labels_data = f_labels['labels']

        # Iterate through the compound label data to extract the defect_depth/_radius/_x_centre field

        for i in range(labels_data.shape[0]):
            label_entry = labels_data[i]
            defect_depth = label_entry['defect_x_centre']  # Extracting only defect_radius
            defect_depth = (round(defect_depth * 1000) / 1000) * 1000  # Converting to fixed-point with 3 decimal places

            # Append corresponding defect_depth to GT
            GT.append(defect_depth)

    # Compute normalization constants
    Y_mean = np.mean(GT)
    Y_max_abs = np.max(np.abs(GT - Y_mean))
    norms = {'YShift': Y_mean, 'YNorm': Y_max_abs}

    # Normalize target labels
    GT_normalized = [(y - Y_mean) / Y_max_abs for y in GT]
    print(signal_data.shape)
    # Iterate over each column of the signal data
    for i in range(signal_data.shape[1]):
        # Process signal to create image
        img, _ = process_file(signal_data[:, i])  # Correct

        # Append image to Arr
        Arr.append(img)

    # Extract the shape of a single processed signal

    single_signal_shape = Arr[0].shape
    print(len(Arr))
    # Return the processed signals, normalized labels, norms, and shape of a single processed signal
    return Arr, GT_normalized, norms, single_signal_shape

########################################################################################################################
# Build Custom Datasets
########################################################################################################################
''' Single Class builder - 3 arrays split to fit 'RBG' of off the shelf neural networks '''

class STFT_CustomDataset(Dataset):
    def __init__(self, stft_data, labels):
        self.stft_data = stft_data
        self.labels = labels
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        stft = self.stft_data[index]  # Get the processed data directly
        stft = np.stack((stft,) * 3, axis=-1)  # Pseudo-RGB layer
        stft = self.transforms(stft)
        label = self.labels[index]
        return stft, torch.tensor(label, dtype=torch.float32)  # Return label as a float tensor

    def __len__(self):
        return len(self.stft_data)

