import os, time
import logging
import random
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from data import Dataset
from graph_conv import GCMCLayer
from utils import get_activation, get_optimizer, torch_total_param_num, \
    torch_net_info, prepare_submission_file, plot_uncertainties, MetricLogger

class Net(nn.Module):
    def __init__(self, args):
        """
        if args.bayesian == False, this class is SVD++ which uses 
        GCN for encoding user-item interactions.
        if args.bayesian == True, then this class is Bayesian version of the
        network mentioned above.
        """
        super(Net, self).__init__()
        self._act = get_activation(args.model_activation)
        self.args = args
        self.encoder_P_Q = GCMCLayer(args.rating_vals,
                                     args.src_in_units,
                                     args.dst_in_units,
                                     args.gcn_agg_units,
                                     args.gcn_out_units * (1 + args.bayesian * 1),
                                     args.gcn_dropout,
                                     args.gcn_agg_accum,
                                     agg_act=self._act,
                                     share_user_item_param=args.share_param,
                                     device=args.device)
        self.encoder_Bu_mu = nn.Embedding(10000, 1)
        nn.init.normal_(self.encoder_Bu_mu.weight, mean=args.mean_init, std=args.std_init)
        self.encoder_Bi_mu = nn.Embedding(1000, 1)
        nn.init.normal_(self.encoder_Bi_mu.weight, mean=args.mean_init, std=args.std_init)
        self.encoder_Y_mu = nn.Embedding(1000 + 1, args.gcn_out_units, padding_idx=0)
        nn.init.normal_(self.encoder_Y_mu.weight, mean=args.mean_init, std=args.std_init)

        if args.bayesian:
            self.encoder_Bu_logsigma = nn.Embedding(10000, 1)
            nn.init.constant_(self.encoder_Bu_logsigma.weight, args.logsigma_constant_init)
            self.encoder_Bi_logsigma = nn.Embedding(1000, 1)
            nn.init.constant_(self.encoder_Bi_logsigma.weight, args.logsigma_constant_init)

    def forward(self, enc_graph, implicit_matrix, sqrt_of_number_of_movies_rated_by_each_user, global_mean, ufeat, ifeat):
        """
        Forward pass of the network.
        """
        p_mu, q_mu = self.encoder_P_Q(
            enc_graph,
            ufeat,
            ifeat)
        bu_mu = self.encoder_Bu_mu.weight
        bi_mu = self.encoder_Bi_mu.weight
        y_mu = self.encoder_Y_mu(implicit_matrix).sum(axis=1).div(sqrt_of_number_of_movies_rated_by_each_user)
        gm = global_mean
        if self.args.bayesian:
            p_mu, p_logsigma = th.split(p_mu, int(p_mu.shape[1] / 2), dim=1)
            p_mu = p_mu + F.softplus(p_logsigma) * th.normal(mean=th.zeros_like(p_mu), std=th.ones_like(p_mu))
            
            q_mu, q_logsigma = th.split(q_mu, int(q_mu.shape[1] / 2), dim=1)
            q_mu = q_mu + F.softplus(q_logsigma) * th.normal(mean=th.zeros_like(q_mu), std=th.ones_like(q_mu))
            
            bu_mu = bu_mu + F.softplus(self.encoder_Bu_logsigma.weight) * th.normal(mean=th.zeros_like(bu_mu), std=th.ones_like(bu_mu))
            bi_mu = bi_mu + F.softplus(self.encoder_Bi_logsigma.weight) * th.normal(mean=th.zeros_like(bi_mu), std=th.ones_like(bi_mu))

        result = q_mu.matmul((p_mu+y_mu).T) + bi_mu + bu_mu.T + gm
        return result
    
    def kl_divergence(self, enc_graph, ufeat, ifeat):
        '''
        Computes the KL divergence between the priors and posteriors of all embeddings.
        '''
        p_mu, q_mu = self.encoder_P_Q(
            enc_graph,
            ufeat,
            ifeat
        )
        p_mu, p_logsigma = th.split(p_mu, int(p_mu.shape[1] / 2), dim=1)
        q_mu, q_logsigma = th.split(q_mu, int(q_mu.shape[1] / 2), dim=1)
        kl_loss = self._kl_divergence(self.encoder_Bu_mu.weight, self.encoder_Bu_logsigma.weight)
        kl_loss += self._kl_divergence(self.encoder_Bi_mu.weight, self.encoder_Bi_logsigma.weight)
        kl_loss += self._kl_divergence(p_mu, p_logsigma)
        kl_loss += self._kl_divergence(q_mu, q_logsigma)
        return kl_loss

    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''
        sigma = F.softplus(logsigma)
        params = mu + sigma * th.normal(mean=th.zeros_like(mu), std=th.ones_like(mu))
        
        p_prior_dist = th.distributions.normal.Normal(self.args.prior_mu, self.args.prior_sigma)
        p_prior_log_prob = p_prior_dist.log_prob(params)
        
        q_posterior_dist = th.distributions.normal.Normal(mu, sigma)
        q_posterior_log_prob = q_posterior_dist.log_prob(params)
        
        kl = th.sum(q_posterior_log_prob - p_prior_log_prob)

        return kl

def evaluate(args, net, dataset, segment='valid'):
    """
    Evaluates the model on the given segment of the dataset.
    Segment is "valid" or "test".
    """
    if segment == "valid":
        labels = dataset.labels
        enc_graph = dataset.valid_enc_graph
        implicit_matrix = dataset.train_implicit_matrix
        sqrt_of_number_of_movies_rated_by_each_user = dataset.train_sqrt_of_number_of_movies_rated_by_each_user
        global_mean = dataset.train_global_mean
        mask = dataset.valid_mask
    elif segment == "test":
        labels = dataset.labels
        enc_graph = dataset.test_enc_graph
        implicit_matrix = dataset.test_implicit_matrix
        sqrt_of_number_of_movies_rated_by_each_user = dataset.test_sqrt_of_number_of_movies_rated_by_each_user
        global_mean = dataset.test_global_mean
        mask = dataset.test_mask
    else:
        raise NotImplementedError

    # Evaluate RMSE
    net.eval()
    with th.no_grad():
        if args.bayesian:
            predictions = None
            # Do forward pass for "num_forward_passes" times and take the mean of the
            # predictions.
            for i in range(args.num_forward_passes):
                current_predictions = net(enc_graph,
                                        implicit_matrix,
                                        sqrt_of_number_of_movies_rated_by_each_user,
                                        global_mean,
                                        dataset.user_feature,
                                        dataset.movie_feature)
                if i == 0:
                    predictions = current_predictions
                else:
                    predictions = (predictions * i + current_predictions) / (i+1)
        else:
            predictions = net(enc_graph,
                              implicit_matrix,
                              sqrt_of_number_of_movies_rated_by_each_user,
                              global_mean,
                              dataset.user_feature,
                              dataset.movie_feature)

        sse = (mask * ((labels - predictions) ** 2)).sum()
        mse = float((sse / mask.sum()).detach().cpu().numpy())
        rmse = np.sqrt(mse)

    return rmse

def train(args):
    """
    Executes the full training pipeline.
    """
    # Set seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    print(args)
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = Dataset(device=args.device, symm=args.gcn_agg_norm_symm,
                      test_ratio=args.data_test_ratio, valid_ratio=args.data_valid_ratio,
                      random_state=args.seed, data_path=args.data_path,
                      make_submission=args.make_submission)
    print("Loading data finished ...\n")

    args.src_in_units = dataset.user_feature_shape[1]
    args.dst_in_units = dataset.movie_feature_shape[1]
    args.rating_vals = dataset.possible_rating_values

    ### Build the net
    net = Net(args=args)
    net = net.to(args.device)
    learning_rate = args.train_lr
    kl_coefficient = args.kl_coefficient
    make_submission = args.make_submission
    if args.bayesian:
        # Don't use weight decay if Bayesian mode is active.
        optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate, weight_decay=0.0)
    else:
        optimizer = get_optimizer(args.train_optimizer)(net.parameters(), lr=learning_rate, weight_decay=args.l2_reg)
    print("Loading network finished ...\n")

    labels = dataset.labels
    mask = dataset.train_mask

    # Create loggers to save loss and rmse information.
    train_loss_logger = MetricLogger(['iter', 'loss', 'rmse'], ['%d', '%.4f', '%.4f'],
                                     os.path.join(args.save_dir, 'train_loss%d.csv' % args.save_id))
    if not make_submission:
        valid_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                        os.path.join(args.save_dir, 'valid_loss%d.csv' % args.save_id))
        test_loss_logger = MetricLogger(['iter', 'rmse'], ['%d', '%.4f'],
                                        os.path.join(args.save_dir, 'test_loss%d.csv' % args.save_id))

    dataset.train_enc_graph = dataset.train_enc_graph.int().to(args.device)
    if not make_submission:
        dataset.valid_enc_graph = dataset.train_enc_graph
        dataset.test_enc_graph = dataset.test_enc_graph.int().to(args.device)

    count_rmse = 0
    count_num = 0
    count_loss = 0
    
    if not make_submission:
        best_valid_rmse = np.inf
        no_better_valid = 0
        best_iter = -1

    print("Start training ...")
    dur = []
    if not make_submission:
        lr_schedule = {}
        kl_coeff_schedule = {}
    else:
        lr_schedule = args.lr_schedule
        kl_coeff_schedule = args.kl_coeff_schedule
    for iter_idx in range(1, args.train_max_iter):
        if iter_idx > 3:
            t0 = time.time()
        net.train()
        if args.bayesian:
            predictions = None
            for i in range(args.num_forward_passes):
                current_predictions = net(dataset.train_enc_graph,
                                        dataset.train_implicit_matrix,
                                        dataset.train_sqrt_of_number_of_movies_rated_by_each_user,
                                        dataset.train_global_mean,
                                        dataset.user_feature,
                                        dataset.movie_feature)
                if i == 0:
                    predictions = current_predictions
                else:
                    predictions = (predictions * i + current_predictions) / (i+1)
        else:
            predictions = net(dataset.train_enc_graph,
                              dataset.train_implicit_matrix,
                              dataset.train_sqrt_of_number_of_movies_rated_by_each_user,
                              dataset.train_global_mean,
                              dataset.user_feature,
                              dataset.movie_feature)
        
        sse = (mask * ((labels - predictions) ** 2)).sum()
        mse = sse / mask.sum()
        
        loss = mse
        if args.bayesian:
            loss += args.l2_reg * th.norm(net.encoder_Y_mu.weight, 2)
            loss += kl_coefficient * net.kl_divergence(dataset.train_enc_graph, dataset.user_feature, dataset.movie_feature)
        count_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.train_grad_clip)
        optimizer.step()
        del predictions

        if iter_idx > 3:
            dur.append(time.time() - t0)

        if iter_idx == 1:
            print("Total #Param of net: %d" % (torch_total_param_num(net)))
            print(torch_net_info(net, save_path=os.path.join(args.save_dir, 'net%d.txt' % args.save_id)))

        count_rmse += float(sse.detach().cpu().numpy())
        count_num += mask.sum()

        if iter_idx % args.train_log_interval == 0:
            train_loss_logger.log(iter=iter_idx,
                                  loss=count_loss/(iter_idx+1), rmse=count_rmse/count_num)
            logging_str = "Iter={}, loss={:.4f}, rmse={:.4f}, time={:.4f}".format(
                iter_idx, count_loss/iter_idx, count_rmse/count_num,
                np.average(dur))
            count_rmse = 0
            count_num = 0

        if make_submission and iter_idx in lr_schedule.keys():
            # Use the learning rate schedule we found during train-validation phase.
            learning_rate = lr_schedule[iter_idx]
            for p in optimizer.param_groups:
                p['lr'] = learning_rate
        
        if make_submission and args.bayesian and iter_idx in kl_coeff_schedule.keys():
            # Use the KL coefficient schedule we found during train-validation phase.
            kl_coefficient = kl_coeff_schedule[iter_idx]

        if not make_submission and iter_idx % args.train_valid_interval == 0:
            # Evaluate model on validation set.
            valid_rmse = evaluate(args=args, net=net, dataset=dataset, segment='valid')
            valid_loss_logger.log(iter = iter_idx, rmse = valid_rmse)
            logging_str += ',\tVal RMSE={:.4f}'.format(valid_rmse)

            if valid_rmse < best_valid_rmse:
                # Evaluate model on test set.
                best_valid_rmse = valid_rmse
                no_better_valid = 0
                best_iter = iter_idx
                test_rmse = evaluate(args=args, net=net, dataset=dataset, segment='test')
                best_test_rmse = test_rmse
                test_loss_logger.log(iter=iter_idx, rmse=test_rmse)
                logging_str += ', Test RMSE={:.4f}'.format(test_rmse)
            else:
                no_better_valid += 1
                if no_better_valid > args.train_early_stopping_patience\
                    and learning_rate <= args.train_min_lr:
                    logging.info("Early stopping threshold reached. Stop training.")
                    break
                if no_better_valid > args.train_decay_patience:
                    if args.bayesian:
                        # Reduce KL coefficient to half.
                        kl_coefficient = kl_coefficient * 0.5
                        kl_coeff_schedule[iter_idx] = kl_coefficient
                    new_lr = max(learning_rate * args.train_lr_decay_factor, args.train_min_lr)
                    if new_lr < learning_rate:
                        # Reduce learning rate.
                        learning_rate = new_lr
                        lr_schedule[iter_idx] = learning_rate
                        logging.info("\tChange the LR to %g" % new_lr)
                        for p in optimizer.param_groups:
                            p['lr'] = learning_rate
                        no_better_valid = 0
        if iter_idx  % args.train_log_interval == 0:
            print(logging_str)
    
    train_loss_logger.close()
    if make_submission:
        # Produce predictions to be saved to disk.
        net.eval()
        if args.bayesian:
            # Do forward pass for args.num_forward_passes times and take their mean.
            # Also calculate standard deviation of predictions so that we can report
            # uncertainties in predictions.
            predictions = []
            with th.no_grad():
                for i in range(args.num_forward_passes):
                    current_predictions = net(dataset.train_enc_graph,
                                              dataset.train_implicit_matrix,
                                              dataset.train_sqrt_of_number_of_movies_rated_by_each_user,
                                              dataset.train_global_mean,
                                              dataset.user_feature,
                                              dataset.movie_feature)
                    predictions.append(current_predictions.cpu().detach().numpy())
            predictions_mean = np.array(predictions)
            predictions_mean = np.mean(predictions, axis=0)
            predictions_std = np.std(predictions, axis=0)

            with th.no_grad():
                # Extract the parameter uncertainties from network.
                Bi_sigma = F.softplus(net.encoder_Bi_logsigma.weight).cpu().detach().numpy()
                Bu_sigma = F.softplus(net.encoder_Bu_logsigma.weight).cpu().detach().numpy()
                p, q = net.encoder_P_Q(dataset.train_enc_graph,
                                    dataset.user_feature,
                                    dataset.movie_feature)
                _, P_logsigma = th.split(p, int(p.shape[1] / 2), dim=1)
                _, Q_logsigma = th.split(q, int(q.shape[1] / 2), dim=1)
                P_sigma = F.softplus(P_logsigma).cpu().detach().numpy()
                Q_sigma = F.softplus(Q_logsigma).cpu().detach().numpy()

            plot_uncertainties(Bi_sigma, Bu_sigma, Q_sigma, P_sigma, predictions_std, args)
            prepare_submission_file(predictions_mean, args)
        else:
            with th.no_grad():
                predictions_mean = net(dataset.train_enc_graph,
                                       dataset.train_implicit_matrix,
                                       dataset.train_sqrt_of_number_of_movies_rated_by_each_user,
                                       dataset.train_global_mean,
                                       dataset.user_feature,
                                       dataset.movie_feature).cpu().detach().numpy()
                prepare_submission_file(predictions_mean, args)
    else:
        print("LR Schedule:", lr_schedule)
        print("KL Schedule:", kl_coeff_schedule)
        print('Best Iter Idx={}, Best Valid RMSE={:.4f}, Best Test RMSE={:.4f}'.format(
            best_iter, best_valid_rmse, best_test_rmse))
        valid_loss_logger.close()
        test_loss_logger.close()
        return best_iter, best_valid_rmse, best_test_rmse, str(lr_schedule), str(kl_coeff_schedule)