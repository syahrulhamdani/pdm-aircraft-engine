import numpy as np
import pandas as pd
import os

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
        """return the remaining useful life for each cycle for each EngineID."""
        num_engine = pd.unique(data.iloc[:, 0]).shape[0]
        num_cycle = [data.loc[data[names[0]]==i, names[0]].shape[0] for i in range(1, num_engine+1)]
        rul = np.array([])
        for engine in range(num_engine):
            diff = num_cycle[engine] - data.loc[data[names[0]]==engine+1, names[1]].values
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