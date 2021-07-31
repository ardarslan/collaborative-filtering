import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rc('axes', labelsize=14)
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class MetricLogger(object):
    """
    Logger class which is used to save loss information.
    """
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        os.makedirs(str(Path(save_path).parent), exist_ok=True)
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()


def torch_total_param_num(net):
    """
    Returns total number of parameters in the network.
    """
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
    """
    Returns number of parameters of the network.
    """
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str


def get_activation(act):
    """Get the activation based on the act string

    Parameters
    ----------
    act: str or callable function

    Returns
    -------
    ret: callable function
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            return nn.LeakyReLU(0.1)
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError
    else:
        return act


def get_optimizer(opt):
    """
    Returns the optimizer which is used during training.
    """
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def prepare_submission_file(predictions, args):
    """
    Writes the predictions file to args.save_dir directory.
    """
    sample_submission = pd.read_csv(f"{args.data_path}/sampleSubmission.csv", engine='python')
    id_column = sample_submission['Id']
    rs = id_column.apply(lambda x: int(x.split("_")[0].split("r")[-1])).tolist()
    cs = id_column.apply(lambda x: int(x.split("_c")[-1])).tolist()
    data = []
    for i, j in zip(rs, cs):
        idx = f"r{i}_c{j}"
        data.append((idx, predictions[j-1, i-1]))
    df = pd.DataFrame(data, columns=['Id', 'Prediction'])
    df.to_csv(f"{args.save_dir}/{args.model}_submission.csv", index=False)


def extract_users_items_labels(data_pd):
    """
    Transforms raw training data into a processed Pandas Dataframe.
    """
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    ratings = data_pd.Prediction.values
    return pd.DataFrame.from_dict({"user_id": users, "movie_id": movies, "rating": ratings})


def plot_uncertainties(Bi_std, Bu_std, Q_std, P_std, preds_std, args):
    """
    Plots uncertainties in model predictions and model parameters. Saves the figures
    to args.save_dir directory.
    """
    Bi_std = Bi_std.ravel()
    Bu_std = Bu_std.ravel()
    Q_std = Q_std.mean(axis=1)
    P_std = P_std.mean(axis=1)
    preds_std_user = preds_std.mean(axis=0)
    preds_std_movie = preds_std.mean(axis=1)

    train_data = extract_users_items_labels(pd.read_csv(f"{args.data_path}/data_train.csv"))

    user_counts = train_data.groupby("user_id").count().reset_index(drop=True)[["movie_id"]].values.ravel()
    movie_counts = train_data.groupby("movie_id").count().reset_index(drop=True)[["user_id"]].values.ravel()

    plt.figure(figsize=(18, 6))
    plt.scatter(user_counts, P_std)
    plt.xlabel("Number of movies rated by the user")
    plt.ylabel("Std of 'P' parameters of the user")
    plt.savefig(f"{args.save_dir}/P_std.png")

    plt.figure(figsize=(18, 6))
    plt.scatter(movie_counts, Q_std)
    plt.xlabel("Number of users who rated the movie")
    plt.ylabel("Std of 'Q' parameters of the movie")
    plt.savefig(f"{args.save_dir}/Q_std.png")

    plt.figure(figsize=(18, 6))
    plt.scatter(user_counts, Bu_std)
    plt.xlabel("Number of movies rated by the user")
    plt.ylabel("Std of 'Bu' parameters of the user")
    plt.savefig(f"{args.save_dir}/Bu_std.png")

    plt.figure(figsize=(18, 6))
    plt.scatter(movie_counts, Bi_std)
    plt.xlabel("Number of users who rated the movie")
    plt.ylabel("Std of 'Bi' parameters of the movie")
    plt.savefig(f"{args.save_dir}/Bi_std.png")

    plt.figure(figsize=(18, 6))
    plt.scatter(user_counts, preds_std_user)
    plt.xlabel("Number of movies rated by the user")
    plt.ylabel("Std of predictions for the user")
    plt.savefig(f"{args.save_dir}/Preds_user_std.png")

    plt.figure(figsize=(18, 6))
    plt.scatter(movie_counts, preds_std_movie)
    plt.xlabel("Number of users who rated the movie")
    plt.ylabel("Std of predictions for the movie")
    plt.savefig(f"{args.save_dir}/Preds_movie_std.png")
