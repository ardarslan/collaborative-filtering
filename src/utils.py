import os
import csv
import re
from pathlib import Path
import torch as th
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class MetricLogger(object):
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
    return sum([np.prod(p.shape) for p in net.parameters()])


def torch_net_info(net, save_path=None):
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
    if opt == 'sgd':
        return optim.SGD
    elif opt == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError


def to_etype_name(rating):
    return str(rating).replace('.', '_')


def prepare_submission_file(predictions, args):
    sample_submission = pd.read_csv(f"{args.data_path}/sampleSubmission.csv", engine='python')
    id_column = sample_submission['Id']
    rs = id_column.apply(lambda x: int(x.split("_")[0].split("r")[-1])).tolist()
    cs = id_column.apply(lambda x: int(x.split("_c")[-1])).tolist()
    data = []
    for i, j in zip(rs, cs):
        idx = f"r{i}_c{j}"
        data.append((idx, predictions[j-1, i-1]))
    df = pd.DataFrame(data, columns=['Id', 'Prediction'])
    df.to_csv(f"{args.save_dir}/{args.save_id}_submission.csv", index=False)


def save_stds(preds_std, Bi_std, Bu_std, P_std, Q_std, args):
    np.save(f"{args.save_dir}/{args.save_id}_preds_std.npy", preds_std)
    np.save(f"{args.save_dir}/{args.save_id}_Bi_std.npy", Bi_std)
    np.save(f"{args.save_dir}/{args.save_id}_Bu_std.npy", Bu_std)
    np.save(f"{args.save_dir}/{args.save_id}_P_std.npy", P_std)
    np.save(f"{args.save_dir}/{args.save_id}_Q_std.npy", Q_std)