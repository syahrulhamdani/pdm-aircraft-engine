import os
from datetime import datetime as dt
from utils.cli import get_validate_arg
from utils.make_dataset import get_processed
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rul_estimation(
    model, signals, label_dict,
    device, save_dir=None
):
    """Estimate RULs for all signal in the given data and create a comparation
    plot between the output and original RUL.

    parameters
    ----------
    model: trained model
    signals: data in DataFrame
    label_dict (dict): label dictionary with cycle length as keys
                        and RULs as value
    device: device to use to do validation
    save_dir (str): Path to save comparison plot

    returns
    -------
    None

    Save line plot of predicted RUL in ``reports``
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        signal = make_tensor(signals)
        signal = signal.type(torch.FloatTensor).to(device)
        print('Estimating RULs..')
        output = model(signal)
        print("[DONE] Estimated!")
    np_output = output.numpy()
    np_output = np_output.reshape(np_output.shape[0]).astype(np.int)
    # convert np_output into dataframe
    concat_output = np.concatenate(
        (signals['EngineID'].values.reshape(signals.shape[0], -1),
         np_output.reshape(np_output.shape[0], -1)), axis=1
    )
    # scaled engine-id for comparison plot
    np_engineid = np.unique(concat_output[:, 0])
    df_output = pd.DataFrame(concat_output, columns=['EngineID', 'Output'])
    print('Create comparison plot..')
    # set directory for saving plot
    date = dt.now().date()
    folder_path = os.path.join(save_dir, dt.strftime(date, '%b-%d-%y'))
    if (save_dir is not None) and (not os.path.isdir(folder_path)):
        print('Create new subfolder..')
        os.mkdir(folder_path)
    # comparison plot of output and original rul for all engines
    for eng in range(1, n_engine+1):
        print('Plot output and RUL engine-{}'.format(eng))
        plt.figure()
        cycle_length = len(
            df_output.loc[df_output['EngineID'] == np_engineid[eng-1]]
        )
        eng_output = df_output.loc[
            df_output['EngineID'] == np_engineid[eng-1], 'Output'
        ]
        plt.plot(
            range(cycle_length),
            eng_output
        )
        plt.plot(range(cycle_length), range(cycle_length-1, -1, -1))
        plt.title(
            'Remaining Useful Life progression of engine-{}'.format(eng)
        )
        plt.legend(['predicted RUL', 'original RUL'])
        plt.ylim(bottom=0)
        plt.xlim(left=0)
        # if engine % 10 == 0:
        #     plt.show()
        plt.savefig(
            os.path.join(
                folder_path, 'Engine-{} rul estimation'.format(eng)
            )
        )
    print('[DONE] Comparison plot is created!')


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
    model.load_state_dict(checkpoint['model_state'])
    optimizer = checkpoint['optim']
    optimizer.load_state_dict(checkpoint['optim_state'])
    epochs = checkpoint['epochs']
    scaler = checkpoint['scaler']

    return model, optimizer, epochs, scaler


def make_tensor(signal):
    tensor = torch.from_numpy(signal.iloc[:, 2:].values)

    return tensor


if __name__ == '__main__':
    argument = get_validate_arg()
    print("Load model..")
    model, o, e, scaler = load_model(argument.model)
    print("[DONE] Loaded!")
    # read processed data
    print('Get processed data..')
    data = get_processed(argument.signal)
    print('[DONE]')
    # define label_dict
    label_dict = {}
    n_engine = data['EngineID'].nunique()
    uniq_engine = np.unique(data['EngineID'])
    for eng in range(1, n_engine+1):
        length = len(
            data.loc[data['EngineID'] == uniq_engine[eng-1], 'EngineID']
        )
        rul = list(range(length-1, -1, -1))
        label_dict[length] = rul
    # estimate the data rul and create comparison plot
    print('Estimate and Compare..')
    rul_estimation(
        model,
        data.iloc[:, :-1],
        label_dict,
        argument.with_gpu,
        save_dir=argument.save_dir
    )
    print('[DONE] Plot saved!')
