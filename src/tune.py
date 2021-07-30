from types import SimpleNamespace
from train import train
import pandas as pd
import random
from itertools import product

model = "bayesian_gc_svdpp" # "bayesian_gc_svdpp" or "gc_svdpp"
device = "cuda" # "cuda" or "cpu"
make_submission = False
data_path = "/cluster/home/aarslan/cil/data"
save_dir = "/cluster/home/aarslan/cil/save_dir"

if model == "bayesian_gc_svdpp":
    bayesian = True
else:
    bayesian = False

train_lr_options = [0.001, 0.01, 0.1, 0.5]
l2_reg_options = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5] # [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
gcn_dropout_options = [0.2, 0.3 , 0.5, 0.7, 0.8]
mean_init_options = [0.001, 0.1, 0.2, 0.3]
gcn_agg_units_options = [200, 400, 600, 800]
gcn_out_units_options = [20, 30, 50, 80]
std_init_options = [0.01, 0.001, 0.0025]
train_grad_clip_options = [1.0, 3.0, 5.0]
gcn_agg_accum_options = ["sum", "stack"]
kl_coefficient_options = [1e-15, 5e-15, 8e-15, 1e-14, 3e-14, 5e-14, 7e-14, 9e-14, 1e-13, 1.5e-13]
prior_mu_options = [0.0]
prior_sigma_options = [0.001, 0.01]
logsigma_constant_init_options = [-8.5, -6.0, -4.5, -3.0]


settings = list(product(train_lr_options, l2_reg_options, gcn_dropout_options, mean_init_options, gcn_agg_units_options, gcn_out_units_options, std_init_options, train_grad_clip_options, gcn_agg_accum_options, kl_coefficient_options, prior_mu_options, prior_sigma_options, logsigma_constant_init_options))
random.shuffle(settings)

results = []
for iter, setting in enumerate(settings):
    train_lr, l2_reg, gcn_dropout, mean_init, gcn_agg_units, gcn_out_units, std_init, train_grad_clip, gcn_agg_accum, bayesian, kl_coefficient, prior_mu, prior_sigma, logsigma_constant_init = setting
    args = SimpleNamespace(
        device=device,
        data_path=data_path,
        save_dir=save_dir,
        save_id=iter,
        silent=False,
        data_test_ratio=0.1,
        data_valid_ratio=0.1,
        model_activation="leaky",
        gcn_dropout=gcn_dropout,
        gcn_agg_norm_symm=True,
        gcn_agg_units=gcn_agg_units,
        gcn_agg_accum=gcn_agg_accum,
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
        lr_schedule=None,
        kl_coeff_schedule=None,
        l2_reg=l2_reg,
        make_submission=make_submission,
        bayesian=bayesian,
        num_forward_passes=5,
        kl_coefficient=kl_coefficient,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
        logsigma_constant_init=logsigma_constant_init
    )

    best_iter, best_valid_rmse, best_test_rmse, lr_schedule, kl_coeff_schedule = train(args)
    results.append((train_lr, l2_reg, gcn_dropout, mean_init,
                    gcn_agg_units, gcn_out_units, std_init,
                    train_grad_clip, gcn_agg_accum,
                    bayesian, kl_coefficient, prior_mu,
                    prior_sigma, logsigma_constant_init,
                    best_iter, best_valid_rmse, best_test_rmse,
                    lr_schedule, kl_coeff_schedule))

    results_df = pd.DataFrame(results,
                              columns=[
                              "train_lr", "l2_reg", "gcn_dropout",
                              "mean_init", "gcn_agg_units", "gcn_out_units",
                              "std_init", "train_grad_clip", "gcn_agg_accum",
                              "bayesian", "kl_coefficient", "prior_mu",
                              "prior_sigma", "logsigma_constant_init",
                              "best_iter", "best_valid_rmse", "best_test_rmse",
                              "lr_schedule", "kl_schedule"]).sort_values(by="best_test_rmse", ascending=False)
    results_df.to_csv(f"{save_dir}/results_{model}.csv", index=False)