import argparse
import torch


def get_argument():
    parser = argparse.ArgumentParser(
        description="argument input for train the network by user"
    )

    parser.add_argument(
        "data", type=str, action='store',
        help="path to dataset", default='train_FD001'
    )
    parser.add_argument(
        "--save_dir", '-save', type=str, action='store',
        help="directory to save the trained model", default='.'
    )
    parser.add_argument(
        '--save_name', '-name', type=str, action='store',
        help="filename of saved model", default='checkpoint.pth'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, action='store',
        help='number of epochs', default=5
    )
    parser.add_argument(
        '--learning_rate', '-lr', action='store', type=float,
        help='learning rate', default=0.001
    )
    parser.add_argument(
        '--hidden_units', '-hu', action='store', nargs='*', type=int,
        help='number and size of hidden layers', default=[32, 24]
    )
    parser.add_argument(
        "--gpu", action='store_true', default=False,
        help='gpu on/off'
    )
    # Add dropout argument input from user
    parser.add_argument(
        "--drop_p", action='store', default=0.4, type=float,
        help="drop out probability"
    )

    arg = parser.parse_args()

    if arg.gpu and torch.cuda.is_available():
        arg.with_gpu = 'cuda'
    else:
        arg.with_gpu = 'cpu'

    return arg


def get_infer_arg():
    parser = argparse.ArgumentParser(
        description="argument input for inference"
    )

    parser.add_argument(
        'signal', action='store', type=str,
        default='test_FD001.txt', help='path to signal test'
    )
    parser.add_argument(
        'model', action='store', type=str, default='checkpoint.pth',
        help='saved model to use for inferencing'
    )
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help='gpu on/off'
    )
    argument = parser.parse_args()

    if argument.gpu and torch.cuda.is_available():
        argument.with_gpu = 'cuda'
    else:
        argument.with_gpu = 'cpu'

    return argument


def get_validate_arg():
    parser = argparse.ArgumentParser(
        description='argument input for validation'
    )

    parser.add_argument(
        'signal', action='store', type=str,
        default='processed.csv', help='path to signal validation'
    )
    parser.add_argument(
        '--gpu', action='store_true', default=False,
        help='gpu on/off'
    )
    parser.add_argument(
        'model', action='store', type=str, default='checkpoint.pth',
        help='saved model to use for validation'
    )
    parser.add_argument(
        "save_dir", type=str, action='store',
        help="directory to save comparation plot",
        default='reports/figures/rul-comparison'
    )
    argument = parser.parse_args()

    if argument.gpu and torch.cuda.is_available():
        argument.with_gpu = 'cuda'
    else:
        argument.with_gpu = 'cpu'

    return argument


if __name__ == '__main__':
    get_argument()
