# --data_name=ml-100k --use_one_hot_fea --gcn_agg_accum=stack

from types import SimpleNamespace
from train import train

args = SimpleNamespace(
    device="cuda", # change to cpu if debugging
    data_path="/cluster/home/aarslan/cil/data/data_train.csv",
    save_dir="/cluster/home/aarslan/cil/gcmc/save_dir",
    save_id=0,
    silent=False,
    data_test_ratio=0.1,
    data_valid_ratio=0.1,
    model_activation="leaky",
    gcn_dropout=0.7,
    gcn_agg_norm_symm=True,
    gcn_agg_units=500,
    gcn_agg_accum="sum", # or "stack"
    gcn_out_units=75,
    gen_r_num_basis_func=2,
    train_max_iter=2000,
    train_log_interval=1,
    train_valid_interval=1,
    train_optimizer="adam",
    train_grad_clip=1.0,
    train_lr=0.01,
    train_min_lr=0.001,
    train_lr_decay_factor=0.5,
    train_decay_patience=50,
    train_early_stopping_patience=100,
    share_param=False,
    seed=42)

train(args)