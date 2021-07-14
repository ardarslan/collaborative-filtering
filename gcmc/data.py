import numpy as np
import os
import re
import pandas as pd
import scipy.sparse as sp
import torch as th

import dgl
from sklearn.model_selection import train_test_split
from utils import to_etype_name


class Dataset(object):
    """Dataset used by GCMC model

    TODO(minjie): make this dataset more general

    The dataset stores ratings in two types of graphs. The encoder graph
    contains rating value information in the form of edge types. The decoder graph
    stores plain user-movie pairs in the form of a bipartite graph with no rating
    information. All graphs have two types of nodes: "user" and "movie".

    The training, validation and test set can be summarized as follows:

    training_enc_graph : training user-movie pairs + rating info
    valid_enc_graph : training user-movie pairs + rating info
    test_enc_graph : training user-movie pairs + validation user-movie pairs + rating info

    Attributes
    ----------
    train_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for training.
    train_labels : torch.Tensor
        The actual rating values of each user-movie pair
    valid_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for validation.
    valid_labels : torch.Tensor
        The actual rating values of each user-movie pair
    test_enc_graph : dgl.DGLHeteroGraph
        Encoder graph for test.
    test_labels : torch.Tensor
        The actual rating values of each user-movie pair
    user_feature : torch.Tensor
        User feature tensor. If None, representing an identity matrix.
    movie_feature : torch.Tensor
        Movie feature tensor. If None, representing an identity matrix.
    possible_rating_values : np.ndarray
        Available rating values in the dataset

    Parameters
    ----------
    name : str
        Dataset name. Could be "ml-100k", "ml-1m", "ml-10m"
    device : torch.device
        Device context
    symm : bool, optional
        If true, the use symmetric normalize constant. Otherwise, use left normalize
        constant. (Default: True)
    test_ratio : float, optional
        Ratio of test data
    valid_ratio : float, optional
        Ratio of validation data

    """
    def __init__(self, device, symm=True,
                 test_ratio=0.1, valid_ratio=0.1,
                 random_state=42, data_path=None):
        self._device = device
        self._symm = symm
        self._test_ratio = test_ratio
        self._valid_ratio = valid_ratio
        self._random_state = random_state
        self._data_path = data_path

        self.all_rating_info = self._extract_users_items_labels(pd.read_csv(self._data_path))
        self.all_train_rating_info, self.test_rating_info = train_test_split(
            self.all_rating_info,
            train_size=1-self._test_ratio,
            random_state=self._random_state
        )

        print('......')
        num_valid = int(np.ceil(self.all_train_rating_info.shape[0] * self._valid_ratio))
        shuffled_idx = np.random.permutation(self.all_train_rating_info.shape[0])
        self.valid_rating_info = self.all_train_rating_info.iloc[shuffled_idx[: num_valid]]
        self.train_rating_info = self.all_train_rating_info.iloc[shuffled_idx[num_valid: ]]
        self.possible_rating_values = np.unique(self.train_rating_info["rating"].values)

        train_movies_rated_by_user_u = {}
        for user, movie in zip(self.train_rating_info.user_id.tolist(), self.train_rating_info.movie_id.tolist()):
            if user in train_movies_rated_by_user_u.keys():
                train_movies_rated_by_user_u[user].append(movie + 1)
            else:
                train_movies_rated_by_user_u[user] = [movie + 1]

        test_movies_rated_by_user_u = {}
        for user, movie in zip(self.train_rating_info.user_id.tolist() + self.valid_rating_info.user_id.tolist(), self.train_rating_info.movie_id.tolist() + self.valid_rating_info.movie_id.tolist()):
            if user in test_movies_rated_by_user_u.keys():
                test_movies_rated_by_user_u[user].append(movie + 1)
            else:
                test_movies_rated_by_user_u[user] = [movie + 1]
        test_largest_number_of_ratings_per_user = max(len(movies) for user, movies in test_movies_rated_by_user_u.items())

        self.train_implicit_matrix = np.zeros(shape=(10000, test_largest_number_of_ratings_per_user))
        for user_id, movies_rated_by_user_u in train_movies_rated_by_user_u.items():
            self.train_implicit_matrix[user_id, :len(movies_rated_by_user_u)] = movies_rated_by_user_u
        self.train_sqrt_of_number_of_movies_rated_by_each_user = th.FloatTensor(np.sqrt(np.count_nonzero(self.train_implicit_matrix, axis=1))).reshape(-1, 1).to(device)
        self.train_implicit_matrix = th.LongTensor(self.train_implicit_matrix).to(device)
        self.train_global_mean = self.train_rating_info.rating.mean()

        self.test_implicit_matrix = np.zeros(shape=(10000, test_largest_number_of_ratings_per_user))
        for user_id, movies_rated_by_user_u in test_movies_rated_by_user_u.items():
            self.test_implicit_matrix[user_id, :len(movies_rated_by_user_u)] = movies_rated_by_user_u
        self.test_sqrt_of_number_of_movies_rated_by_each_user = th.FloatTensor(np.sqrt(np.count_nonzero(self.test_implicit_matrix, axis=1))).reshape(-1, 1).to(device)
        self.test_implicit_matrix = th.LongTensor(self.test_implicit_matrix).to(device)
        self.test_global_mean = np.hstack((self.train_rating_info.rating.values, self.valid_rating_info.rating.values)).mean()

        print("All rating pairs : {}".format(self.all_rating_info.shape[0]))
        print("\tAll train rating pairs : {}".format(self.all_train_rating_info.shape[0]))
        print("\t\tTrain rating pairs : {}".format(self.train_rating_info.shape[0]))
        print("\t\tValid rating pairs : {}".format(self.valid_rating_info.shape[0]))
        print("\tTest rating pairs  : {}".format(self.test_rating_info.shape[0]))

        self.global_user_id_map = {ele: i for i, ele in enumerate(self.all_rating_info['user_id'].unique().tolist())}
        self.global_movie_id_map = {ele: i for i, ele in enumerate(self.all_rating_info['movie_id'].unique().tolist())}
        print('Total user number = {}, movie number = {}'.format(len(self.global_user_id_map),
                                                                 len(self.global_movie_id_map)))
        self._num_user = len(self.global_user_id_map)
        self._num_movie = len(self.global_movie_id_map)

        self.user_feature = None
        self.movie_feature = None
        self.user_feature_shape = (self.num_user, self.num_user)
        self.movie_feature_shape = (self.num_movie, self.num_movie)
        
        info_line = "Feature dim: "
        info_line += "\nuser: {}".format(self.user_feature_shape)
        info_line += "\nmovie: {}".format(self.movie_feature_shape)
        print(info_line)

        all_train_rating_pairs, all_train_rating_values = self._generate_pair_value(self.all_train_rating_info)
        train_rating_pairs, train_rating_values = self._generate_pair_value(self.train_rating_info)
        valid_rating_pairs, valid_rating_values = self._generate_pair_value(self.valid_rating_info)
        test_rating_pairs, test_rating_values = self._generate_pair_value(self.test_rating_info)

        self.train_enc_graph = self._generate_enc_graph(train_rating_pairs, train_rating_values, add_support=True)
        self.valid_enc_graph = self.train_enc_graph
        self.test_enc_graph = self._generate_enc_graph(all_train_rating_pairs, all_train_rating_values, add_support=True)

        self.labels = np.zeros(shape=(1000, 10000))
        self.train_mask = np.zeros(shape=(1000, 10000))
        self.valid_mask = np.zeros(shape=(1000, 10000))
        self.test_mask = np.zeros(shape=(1000, 10000))
        for train_user_id, train_movie_id, train_rating in zip(train_rating_pairs[0], train_rating_pairs[1], train_rating_values):
            self.labels[train_movie_id, train_user_id] = train_rating
            self.train_mask[train_movie_id, train_user_id] = 1
            
        for valid_user_id, valid_movie_id, valid_rating in zip(valid_rating_pairs[0], valid_rating_pairs[1], valid_rating_values):
            self.labels[valid_movie_id, valid_user_id] = valid_rating
            self.valid_mask[valid_movie_id, valid_user_id] = 1

        for test_user_id, test_movie_id, test_rating in zip(test_rating_pairs[0], test_rating_pairs[1], test_rating_values):
            self.labels[test_movie_id, test_user_id] = test_rating
            self.test_mask[test_movie_id, test_user_id] = 1
        
        self.labels = th.IntTensor(self.labels).to(device)
        self.train_mask = th.IntTensor(self.train_mask).to(device)
        self.valid_mask = th.IntTensor(self.valid_mask).to(device)
        self.test_mask = th.IntTensor(self.test_mask).to(device)

        def _npairs(graph):
            rst = 0
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                rst += graph.number_of_edges(str(r))
            return rst

        print("Train enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.train_enc_graph.number_of_nodes('user'), self.train_enc_graph.number_of_nodes('movie'),
            _npairs(self.train_enc_graph)))
        print("Valid enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.valid_enc_graph.number_of_nodes('user'), self.valid_enc_graph.number_of_nodes('movie'),
            _npairs(self.valid_enc_graph)))
        print("Test enc graph: \t#user:{}\t#movie:{}\t#pairs:{}".format(
            self.test_enc_graph.number_of_nodes('user'), self.test_enc_graph.number_of_nodes('movie'),
            _npairs(self.test_enc_graph)))

    def _extract_users_items_labels(self, data_pd):
        users, movies = \
            [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
        ratings = data_pd.Prediction.values
        return pd.DataFrame.from_dict({"user_id": users, "movie_id": movies, "rating": ratings})

    def _generate_pair_value(self, rating_info):
        rating_pairs = (np.array([self.global_user_id_map[ele] for ele in rating_info["user_id"]],
                                 dtype=np.int64),
                        np.array([self.global_movie_id_map[ele] for ele in rating_info["movie_id"]],
                                 dtype=np.int64))
        rating_values = rating_info["rating"].values.astype(np.float32)
        return rating_pairs, rating_values

    def _generate_enc_graph(self, rating_pairs, rating_values, add_support=False):
        user_movie_R = np.zeros((self._num_user, self._num_movie), dtype=np.float32)
        user_movie_R[rating_pairs] = rating_values

        data_dict = dict()
        num_nodes_dict = {'user': self._num_user, 'movie': self._num_movie}
        rating_row, rating_col = rating_pairs
        for rating in self.possible_rating_values:
            ridx = np.where(rating_values == rating)
            rrow = rating_row[ridx]
            rcol = rating_col[ridx]
            rating = to_etype_name(rating)
            data_dict.update({
                ('user', str(rating), 'movie'): (rrow, rcol),
                ('movie', 'rev-%s' % str(rating), 'user'): (rcol, rrow)
            })
        graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

        # sanity check
        assert len(rating_pairs[0]) == sum([graph.number_of_edges(et) for et in graph.etypes]) // 2

        if add_support:
            def _calc_norm(x):
                x = x.numpy().astype('float32')
                x[x == 0.] = np.inf
                x = th.FloatTensor(1. / np.sqrt(x))
                return x.unsqueeze(1)
            user_ci = []
            user_cj = []
            movie_ci = []
            movie_cj = []
            for r in self.possible_rating_values:
                r = to_etype_name(r)
                user_ci.append(graph['rev-%s' % r].in_degrees())
                movie_ci.append(graph[r].in_degrees())
                if self._symm:
                    user_cj.append(graph[r].out_degrees())
                    movie_cj.append(graph['rev-%s' % r].out_degrees())
                else:
                    user_cj.append(th.zeros((self.num_user,)))
                    movie_cj.append(th.zeros((self.num_movie,)))
            user_ci = _calc_norm(sum(user_ci))
            movie_ci = _calc_norm(sum(movie_ci))
            if self._symm:
                user_cj = _calc_norm(sum(user_cj))
                movie_cj = _calc_norm(sum(movie_cj))
            else:
                user_cj = th.ones(self.num_user,)
                movie_cj = th.ones(self.num_movie,)
            graph.nodes['user'].data.update({'ci' : user_ci, 'cj' : user_cj})
            graph.nodes['movie'].data.update({'ci' : movie_ci, 'cj' : movie_cj})

        return graph

    @property
    def num_links(self):
        return self.possible_rating_values.size

    @property
    def num_user(self):
        return self._num_user

    @property
    def num_movie(self):
        return self._num_movie
