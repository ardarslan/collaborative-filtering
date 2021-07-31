from types import SimpleNamespace
from train import train


# To run the code make these 4 configurations:

model = "bayesian_gc_svdpp" # "bayesian_gc_svdpp" or "gc_svdpp"
device = "cuda" # "cuda" or "cpu"
data_path = "../data" # Data path
save_dir = "../save_dir" # Path where the outputs will be saved to.

assert model in ["bayesian_gc_svdpp", "gc_svdpp"]
assert device in ["cpu", "cuda"]

if model == "bayesian_gc_svdpp":
    train_lr = 0.01
    l2_reg = 1e-09
    gcn_dropout = 0.7
    mean_init = 0.1
    gcn_agg_units = 400
    gcn_out_units = 30
    std_init = 0.0025
    train_grad_clip = 3.0
    gcn_agg_accum = "sum"
    bayesian = True
    kl_coefficient = 5e-14
    prior_mu = 0.0
    prior_sigma = 0.001
    num_forward_passes = 5
    logsigma_constant_init = -8.5
    train_max_iter = 624
    lr_schedule = {137: 0.0075, 599: 0.005625, 675: 0.00421875, 726: 0.0031640625, 777: 0.002373046875, 828: 0.0017797851562500002, 879: 0.0013348388671875003}
    kl_coeff_schedule = {137: 2.5e-14, 599: 1.25e-14, 675: 6.25e-15, 726: 3.125e-15, 777: 1.5625e-15, 828: 7.8125e-16, 879: 3.90625e-16}
    save_id = 0
elif model == "gc_svdpp":
    train_lr = 0.5
    l2_reg = 5e-5
    gcn_dropout = 0.3
    mean_init = 0.3
    gcn_agg_units = 500
    gcn_out_units = 40
    std_init = 0.01
    train_grad_clip = 3.0
    gcn_agg_accum = "sum"
    bayesian = False
    kl_coefficient = None
    prior_mu = None
    prior_sigma = None
    num_forward_passes = None
    logsigma_constant_init = None
    train_max_iter = 1193
    lr_schedule = {94: 0.375, 271: 0.28125, 322: 0.2109375, 384: 0.158203125, 550: 0.11865234375, 610: 0.0889892578125, 787: 0.066741943359375, 946: 0.05005645751953125, 1085: 0.03754234313964844, 1244: 0.03125}
    kl_coeff_schedule = None
    save_id = 1

args = SimpleNamespace(
    model=model,
    device=device,
    data_path=data_path,
    save_dir=save_dir,
    save_id=save_id,
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
    train_max_iter=train_max_iter,
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
    lr_schedule=lr_schedule,
    kl_coeff_schedule=kl_coeff_schedule,
    l2_reg=l2_reg,
    make_submission=True,
    bayesian=bayesian,
    num_forward_passes=num_forward_passes,
    kl_coefficient=kl_coefficient,
    prior_mu=prior_mu,
    prior_sigma=prior_sigma,
    logsigma_constant_init=logsigma_constant_init
)

train(args)