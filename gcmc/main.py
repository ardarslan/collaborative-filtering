from types import SimpleNamespace
from train import train
import pandas as pd
import random
from itertools import product


train_lr_options = [5e-4, 5e-3, 5e-2, 5e-1]
l2_reg_options = [5e-6, 1e-5, 5e-5, 1e-4]
gcn_dropout_options = [0.3, 0.5, 0.7]
mean_init_options = [0.1, 0.2, 0.3]
gcn_agg_units_options = [250, 500, 1000]
gcn_out_units_options = [40, 75, 110]
std_init_options = [1e-4, 1e-3, 1e-2]
train_grad_clip_options = [1.0, 2.0, 3.0]
gcn_agg_accum_options = ["stack", "sum"]

settings = list(product(train_lr_options, l2_reg_options, gcn_dropout_options, mean_init_options, gcn_agg_units_options, gcn_out_units_options, std_init_options, train_grad_clip_options, gcn_agg_accum_options))
random.shuffle(settings)

results = []
for iter, setting in enumerate(settings):
    train_lr, l2_reg, gcn_dropout, mean_init, gcn_agg_units, gcn_out_units, std_init, train_grad_clip, gcn_agg_accum = setting
    args = SimpleNamespace(
        device="cuda", # change to cpu if debugging
        data_path="/cluster/home/aarslan/cil/data/data_train.csv",
        save_dir="/cluster/home/aarslan/cil/gcmc/save_dir",
        save_id=iter,
        silent=False,
        data_test_ratio=0.1,
        data_valid_ratio=0.1,
        model_activation="leaky",
        gcn_dropout=gcn_dropout,
        gcn_agg_norm_symm=True,
        gcn_agg_units=gcn_agg_units,
        gcn_agg_accum=gcn_agg_accum, # or "stack"
        gcn_out_units=gcn_out_units,
        gen_r_num_basis_func=2,
        train_max_iter=10000,
        train_log_interval=1,
        train_valid_interval=1,
        train_optimizer="adam",
        train_grad_clip=train_grad_clip,
        train_lr=train_lr,
        train_min_lr=train_lr/16.0,
        train_lr_decay_factor=0.75,
        train_decay_patience=50,
        train_early_stopping_patience=100,
        share_param=False,
        mean_init=mean_init,
        std_init=std_init,
        seed=42,
        l2_reg=l2_reg)

    best_iter, best_valid_rmse, best_test_rmse = train(args)
    results.append((train_lr, l2_reg, gcn_dropout, mean_init,
                    gcn_agg_units, gcn_out_units, std_init,
                    train_grad_clip, gcn_agg_accum,
                    best_iter, best_valid_rmse, best_test_rmse))

    results_df = pd.DataFrame(results,
                                columns=[
                                "train_lr", "l2_reg", "gcn_dropout",
                                "mean_init", "gcn_agg_units", "gcn_out_units",
                                "std_init", "train_grad_clip", "gcn_agg_accum",
                                "best_iter", "best_valid_rmse", "best_test_rmse"
                                ]).sort_values(by="best_test_rmse", ascending=False)
    results_df.to_csv("/cluster/home/aarslan/cil/gcmc/save_dir/results.csv", index=False)