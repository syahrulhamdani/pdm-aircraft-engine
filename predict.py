import torch
import matplotlib.pyplot as plt
from src.cli import get_infer_arg
import pandas as pd


def signal_prep(signal, scaler):
    """preprocess test signal by doing standardization.

    parameters
    ----------
    signal (array-like): signal to be inferred
    scaler (scaler-object): Scaler to standardize the signal

    returns
    -------
    scaled_signal (array-like): scaled signal with the same shape as ``signal``
    """
    pass


def infer(model, data):
    pass


def load_model(model_dir):
    """Load saved model.

    parameters
    ----------
    model_dir (str): directory where saved model exist

    returns
    -------
    model: loaded model
    optimizer: optimizer of pytorch
    epochs: epochs
    """
    checkpoint = torch.load(model_dir)
    model = checkpoint['model']
    model.load_state(checkpoint['model_state'])
    optimizer = checkpoint['optimizer']
    optimizer.load_state(checkpoint['optim_state'])
    epochs = checkpoint['epochs']
    scaler = checkpoint['scaler']

    return model, optimizer, epochs, scaler


if __name__ == '__main__':
    # get user input
    argument = get_infer_arg()
    # load the model
    print('Loading saved model..')
    model, optimizer, epochs, sclaer = load_model(argument.model)
    print('[DONE] Loaded!')
    # preprocess the signal
    print('Preprocess test signal of {}..'.format(argument.signal))
    scaled_signal = signal_prep(argument.signal, argument.signal)
    print('[DONE] {} is preprocessed successfully!'.format(argument.signal))
