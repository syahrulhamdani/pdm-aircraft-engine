import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import argparse
from datetime import datetime as dt
import json


def get_user_input():
    """Get user input only for `make_dataset`."""
    parser = argparse.ArgumentParser(
        description="user input for `make_dataset.py`"
    )
    parser.add_argument(
        'data', type=str, action='store', help='data filename',
        default='train_FD001.txt'
    )
    parser.add_argument(
        '--save_interim', '-si', action='store_true',
        default=False, help='save interim data or not'
    )
    argument = parser.parse_args()

    return argument


def get_json(filename):
    """Return features title from json file."""
    with open(filename, 'r') as f:
        featurelist = json.load(f)

    return list(featurelist.values())


def list_dataset(path='data/raw'):
    """return list of dataset exist in the `path`."""
    key_to_file = dict()
    for directory, _, files in os.walk('data/raw/'):
        for file in files:
            if file.endswith('.txt'):
                key_to_file[file.split('.')[0]] = file

    return key_to_file


class LoadData:
    key_to_file = dict()
    for directory, _, files in os.walk('data/raw/'):
        for file in files:
            if file.endswith('.txt'):
                key_to_file[file.split('.')[0]] = file

    def __init__(self, dict_to_file, folder='data/raw', names=None, sep='\s+'):
        """Load the dataset with name `filename` from `path`.

        parameters
        ----------
        filename (str): the name of dataset exist in path
        folder (str): directory where the data exist
        names (list of str): list of feature names
        sep (regex-string): separator for loading the dataset using pandas

        attributes
        ----------
        features: data features
        target: data labels
        """
        # load the data
        file = os.path.join(folder, dict_to_file)
        dataset = pd.read_csv(file, sep=sep, names=names)
        self.features = dataset.values
        self.target = self.__get_rul(dataset, names)

    def __get_rul(self, data, names):
        """return the remaining useful life for each cycle
        for each EngineID."""
        num_engine = pd.unique(data.iloc[:, 0]).shape[0]
        num_cycle = [
            data.loc[data[names[0]] == i, names[0]].shape[0] for i in range(
                1, num_engine + 1)
        ]
        rul = np.array([])
        for engine in range(num_engine):
            diff = num_cycle[engine] - data.loc[
                    data[names[0]] == (engine+1), names[1]
                    ].values
            rul = np.append(rul, diff)

        return rul

    def save_interim(self, path, names):
        """Save interim data."""
        interim = np.concatenate(
            (self.features, self.target.reshape(self.target.shape[0], -1)),
            axis=1
        )
        np.savetxt(
            os.path.join(path, 'interim.csv'), interim, fmt='%.3f',
            delimiter=',', header=','.join(names), comments=''
        )

    def standardize(self):
        """Standardize features in ``data.features``.

        returns
        -------
        standardized: standardized features
        """
        scaler = StandardScaler()
        standardized = scaler.fit_transform(self.features)

        return standardized, scaler


def get_processed(filename):
    """Read processed data from ``path``."""
    filename = os.path.join('data/processed', filename)
    processed_data = pd.read_csv(filename)

    return processed_data


if __name__ == '__main__':
    # get user niput
    argument = get_user_input()
    # get features title from `col_to_feat` in `references`
    feat_name = get_json('references/col_to_feat.json')
    # load raw data
    raw_data = LoadData(argument.data, names=feat_name, sep='\s+')
    # standardize the data
    scaled_data, scaler = raw_data.standardize()
    # concatenate `target` into `features`
    processed_data = np.concatenate((
        scaled_data, raw_data.target.reshape(
             raw_data.target.shape[0], -1
        )),
        axis=1
    )
    # save `processed_data` and `interim` data (optional)
    date = dt.now().date()
    if argument.save_interim:
        print('Save interim data first..')
        prefix = argument.data.split('.')[0]
        filename = \
            'data/interim/{}_'.format(prefix) \
            + dt.strftime(date, '%b-%d-%y') \
            + '_i.csv'
        np.savetxt(
            filename, scaled_data, delimiter=',',
            header=','.join(feat_name), fmt='%.3f',
            comments=''
        )
    feat_name = np.append(feat_name, 'RUL')
    print('Save processed data..')
    prefix = argument.data.split('.')[0]
    filename = \
        'data/processed/{}'.format(prefix) \
        + dt.strftime(date, '%b-%d-%y') \
        + '_p.csv'
    np.savetxt(
        filename, processed_data, delimiter=',',
        header=','.join(feat_name), fmt='%.3f',
        comments=''
    )
    print("[DONE] All data's saved!")
