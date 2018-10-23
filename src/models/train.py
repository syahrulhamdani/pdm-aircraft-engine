import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from src.cli import get_argument
from src.models import NN
from src.data import LoadData, get_json


def training(
    model,
    criterion,
    optimizer,
    featureloader,
    labelloader,
    epochs=15,
    print_every=40
):
    epoch_loss = 0
    steps = 0

    for epoch in epochs:
        for features, labels in zip(featureloader, labelloader):
            steps += 1
            features = features.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)
            labels.resize_(labels.shape[0], 1)

            output = model.forward(features)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if steps % print_every == 0:
                print('Epoch: {}/{}..'.format(epoch, epochs),
                      'Training loss: {:.3f}'.format(epoch_loss/print_every))

                epoch_loss = 0


if __name__ == '__main__':
    argument = get_argument()
    feature_name = get_json('../../references/col_to_feat.json')
    train = LoadData(argument.data, names=feature_name, sep='\s+')
    scaled_train = train.standardize()
    # convert dataset into tensor
    featureset = torch.from_numpy(scaled_train[:, 2:])
    labelset = torch.from_numpy(train.target)
    # define data loader
    featureloader = torch.utils.data.DataLoader(featureset, batch_size=32)
    labelloader = torch.utils.data.DataLoader(labelset, batch_size=32)
    # define the model
    model = NN(hidden_sizes=argument.hidden_units, drop_p=argument.drop_p)
    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=argument.learning_rate)
    # train the model
    training(
        model, criterion, optimizer,
        featureloader, labelloader,
        epochs=argument.epochs
    )
