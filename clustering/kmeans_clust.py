# -*- coding: utf-8 -*-
"""collaborative_filtering.ipynb

# https://arxiv.org/pdf/1706.02263.pdf graph conv

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/dalab/lecture_cil_public/blob/master/exercises/2021/Project_1.ipynb

# Project 1 - Collaborative Filtering
### ETH Computational Intelligence Lab 2021 - Project 1 

We are given a set of N = 1,176,952$ integer movie ratings, ranging from 1 to 5, 
that are assigned by m=10,000 users to n=1,000 movies. 
A rating $r_{ui}$ indicates the preference by user $u$ of item $i$. 
Let $\mathcal{\Omega} = \{(u,i) : r_{ui} \text{ is known} \}$ be the set of user and movie indices for which the ratings are known.
"""
#@title Basic Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

import pickle

# clustering imports
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


number_of_users, number_of_movies = (10000, 1000)

data_pd = pd.read_csv('data_train.csv') # TODO where will we put the data at submission time?
print(data_pd.head(5))
print()
print('Shape', data_pd.shape)

"""The provided data $\mathcal{\Omega}$ are split into two disjoint subsets, namely $\mathcal{\Omega}_{\text{train}}$ and $\mathcal{\Omega}_{\text{test}}$. The former consists of $90\%$ of the data and is used for training the individual models whereas the latter consists of the remaining $10\%$ of the data and is used for learning optimal blending weights. Depending on your method, you may choose to retrain on the whole dataset for the final solution."""

from sklearn.model_selection import train_test_split
# Split the dataset into train and test

train_size = 0.9

train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)
print('train shape', train_pd.shape)
print(train_pd.head(5))
print('test shape', test_pd.shape)
print(test_pd.head(5))

"""Preprocess data by creating a $m \times n$ matrix

$$A_{ui} = \begin{cases} 
      r_{ui} & \text{ if } (u,i) \in \mathcal{\Omega}_{\text{train}} \\
      0 & \text{ else }
\end{cases}.$$
"""

def extract_users_items_predictions(data_pd):
    users, movies = \
        [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
assert(min(train_users) >= 0 and max(train_users) < number_of_users), "extract_users_items_predictions"


print("training stuff", train_users.shape)
print(train_users[0])
print("training stuff", train_movies.shape)
print(train_movies[0])
print("training stuff", train_predictions.shape)
print(train_predictions[0])

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
data = np.zeros((number_of_users, number_of_movies)) # 
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, train_predictions):
    data[user ][movie ] = pred
    mask[user ][movie ] = 1
print('data', data.shape)

"""To consider:
* Do unobserved values matter for the method we are using? If yes, is the above initialization the best?
* Does normalization of the data matter for the method we are using?
* If yes, should we do the same transformations for the test data?
"""

"""
Our task is to predict ratings according to specific users-movies combinations. We will quantify the quality of our predictions based on the root mean squared error (RMSE) function between the true and observed ratings. For a given set of observations $ \mathcal{\Omega}$, let
\begin{equation}
    \text{RMSE} = \sqrt{\frac{1}{|\mathcal{\Omega}|}\sum_{ (u,i) \in \mathcal{\Omega}} \big(r_{ui} - \hat{r}_{ui}\big)^2}
\end{equation}
where $\hat{r}_{ui}$ denotes the estimate of $r_{ui}$.
"""



rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

# test our predictions with the true values
def get_score(predictions, target_values=test_predictions):
    return rmse(predictions, target_values)

def extract_prediction_from_full_matrix(reconstructed_matrix, users=test_users, movies=test_movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(test_users))

    print("min users ", min(users))
    print("max users ", max(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        assert(i >= 0 and user >= 0 and movie >= 0), "extract_prediction_from_full_matrix"
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions

# ------------------
print("min train users: ", min(train_users) )
print("max train users: ", max(train_users) )

print("min test users: ", min(test_users) )
print("max test users: ", max(test_users) )



# ==================
# calculate avg user ratings (TODO: adding into loop of create full matrix might improve performance)
summed_ratings_per_user = np.zeros(number_of_users)
num_ratings_per_user = np.zeros(number_of_users)
for user, movie, pred in zip(train_users, train_movies, train_predictions):
    summed_ratings_per_user[user - 1] += pred
    num_ratings_per_user[user - 1] += 1
avg_ratings_per_user = np.divide(summed_ratings_per_user, num_ratings_per_user, where = num_ratings_per_user != 0)
"""
print('avgs', avg_ratings_per_user.shape)
print('sums', summed_ratings_per_user[:630])
print('num', num_ratings_per_user[:630])
print('avgs', avg_ratings_per_user[:630])
"""
# calculate avg movie ratings 
summed_ratings_per_movie = np.zeros(number_of_movies)
num_ratings_per_movie = np.zeros(number_of_movies)
for user, movie, pred in zip(train_users, train_movies, train_predictions):
    summed_ratings_per_movie[movie - 1] += pred
    num_ratings_per_movie[movie - 1] += 1
avg_ratings_per_movie = np.divide(summed_ratings_per_movie, num_ratings_per_movie, where = num_ratings_per_movie != 0)

# use per user AND per movie in case we later want to switch between user-to-user and item-to-item collaborative filtering


# TODO for above use np mean


# KMeans to cluster data into 10 clusters of users
"""
num_clusters_users = 10
kmeans_users = KMeans(init="k-means++", n_clusters=num_clusters_users, n_init=4, random_state=0).fit(data)
# save model
filename = "kmeans_users_" + str(num_clusters_users) + ".sav"
pickle.dump(kmeans_users, open(filename, 'wb'))
user_labels = kmeans_users.labels_

# KMeans to cluster data into 10 clusters of movies
num_clusters_movies = 20
kmeans_movies = KMeans(init="k-means++", n_clusters=num_clusters_movies, n_init=4, random_state=0).fit(np.transpose(data))
# save model
filename = "kmeans_movies_" + str(num_clusters_movies) + ".sav"
pickle.dump(kmeans_movies, open(filename, 'wb'))
movie_labels = kmeans_movies.labels_
"""

# ------------------
"""
# load model
print("kmeans")
filename = 'trainedKMeans.sav'
kmeans = pickle.load(open(filename, 'rb'))
user_labels = kmeans.labels_
"""
"""
# trying DBSCAN baseline
data_norm = StandardScaler().fit_transform(data) # TODO: do we want to normalize?
db = DBSCAN(eps=10, min_samples=10).fit(data_norm)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


clust = OPTICS(min_samples=50, xi=.05, min_cluster_size=.05)# TODO: for metric -> important how to treat unrated movies (right now rating=0)
clust.fit(data_norm)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)
labels = clust.labels_[clust.ordering_]
print('clust', labels)

#print(clust.labels)
"""

# ------------------
# make predictor based on ratings of members of cluster (that rated the movie)
# Assumption: unobserved values == 0
# -> for each movie calculate average of cluster


# create list per cluster containing users/movies in cluster
def get_cluster_list(num_clusters, labels):
    cluster_list = []
    for l in range(num_clusters):
        cluster_list.append(np.where(labels == l)[0])
    return cluster_list
# usage
#cluster_list_users = get_cluster_list(num_clusters_users, user_labels)
#cluster_list_movies = get_cluster_list(num_clusters_movies, movie_labels)

# cluster == label
def get_avg_of_clusters(clusters_users, clusters_movies):
    avg_user_cluster = []
    for lu in clusters_users:
        selected_data = data[lu[:], :]
        temp_num_ratings = (selected_data>0.0001).astype(int) #
        temp_num = np.sum(temp_num_ratings, axis=0)  
        temp_sum = np.sum(selected_data, axis=0)  
        avg_user_cluster.append(np.divide(temp_sum, temp_num, where = temp_num != 0))

    avg_movie_cluster = []
    for lm in clusters_movies:
        selected_data = np.transpose(data)[lm[:], :]
        temp_num_ratings = (selected_data>0.0001).astype(int) #
        temp_num = np.sum(temp_num_ratings, axis=0)  
        temp_sum = np.sum(selected_data, axis=0)  
        avg_movie_cluster.append(np.divide(temp_sum, temp_num, where = temp_num != 0))

    return avg_user_cluster, avg_movie_cluster
# usage
#avg_rat_user_cluster, avg_rat_movie_cluster = get_avg_of_clusters(cluster_list_users, cluster_list_movies)

def get_avg_of_user_clust(user, movie):
    return avg_rat_user_cluster[user_labels[user]][movie]
def get_avg_of_movie_clust(user, movie):
    return avg_rat_movie_cluster[movie_labels[movie]][user]

def sim_movie_avg(clusters_movies):
    sim_mvs = []
    for user in range(number_of_users):
        clust_mv_u = [] # per user contains rated movies per cluster 
        for lm in clusters_movies:
            lm_u = []
            for v in lm:
                if mask[user][v]:
                    lm_u.append(v)
            lm_u = np.array(lm_u, dtype=np.int64)
            clust_mv_u.append(lm_u)

        avg_movie_cluster = [] #########
        for lm in clust_mv_u:

            selected_data = data[user][lm]

            temp_num_ratings = (selected_data>0.0001).astype(int) #
            temp_num = np.sum(temp_num_ratings, axis=0)  
            temp_sum = np.sum(selected_data, axis=0)  
            avg_movie_cluster.append(np.divide(temp_sum, temp_num, where = temp_num != 0))

        sim_mvs.append(avg_movie_cluster) 


    return sim_mvs


# Predictor
# set unobserved values to avg of cluster
def predict_data(num_clusters, sim_mvs, rsme_kmeans_users, rsme_kmeans_movies, rsme_kmeans_simmvs):
    reconstructed_matrix_users = data.copy()
    count = 0
    for user in range(number_of_users):
        for movie in range(number_of_movies):
            if reconstructed_matrix_users[user][movie] == 0:
                if(get_avg_of_user_clust(user, movie) > 0.001):
                    reconstructed_matrix_users[user][movie] = get_avg_of_user_clust(user, movie)
                    count = count +1
                else:
                    reconstructed_matrix_users[user][movie] = np.mean(train_pd.Prediction.values)

    predictions = extract_prediction_from_full_matrix(reconstructed_matrix_users)
    print(predictions.shape)

    print("RMSE using kmeans is: {:.4f}".format(get_score(predictions)))
    print("count", count)
    rsme_kmeans_users.append((num_clusters, get_score(predictions)))
    # compute_svd(num_clusters, reconstructed_matrix_users, rsme_kmeans_users_svd )
    # ------------------
    reconstructed_matrix_movies = data.copy()
    count = 0
    for user in range(number_of_users):
        for movie in range(number_of_movies):
            if reconstructed_matrix_movies[user][movie] == 0:
                if(get_avg_of_movie_clust(user, movie) > 0.001):
                    reconstructed_matrix_movies[user][movie] = get_avg_of_movie_clust(user, movie)
                    count = count +1
                else:
                    reconstructed_matrix_movies[user][movie] = np.mean(train_pd.Prediction.values)


    predictions = extract_prediction_from_full_matrix(reconstructed_matrix_movies)
    print(predictions.shape)

    print("RMSE using kmeans is: {:.4f}".format(get_score(predictions)))
    print("count", count)
    rsme_kmeans_movies.append((num_clusters, get_score(predictions)))

    # compute_svd(num_clusters, reconstructed_matrix_movies, rsme_kmeans_movies_svd )
    # ------------------
    reconstructed_matrix_simmvs = data.copy()
    count = 0
    for user in range(number_of_users):
        for movie in range(number_of_movies):
            if reconstructed_matrix_simmvs[user][movie] == 0:
                if(sim_mvs[user][movie_labels[movie]]):
                    reconstructed_matrix_simmvs[user][movie] = sim_mvs[user][movie_labels[movie]]
                    count = count +1
                else:
                    if(get_avg_of_user_clust(user, movie) > 0.001):
                        reconstructed_matrix_users[user][movie] = get_avg_of_user_clust(user, movie)
                    else:
                        reconstructed_matrix_users[user][movie] = np.mean(train_pd.Prediction.values)


    predictions = extract_prediction_from_full_matrix(reconstructed_matrix_simmvs)
    print(predictions.shape)

    print("RMSE using kmeans is: {:.4f}".format(get_score(predictions)))
    print("count", count)
    rsme_kmeans_simmvs.append((num_clusters, get_score(predictions)))
    # compute_svd(num_clusters, reconstructed_matrix_simmvs, rsme_kmeans_simmvs_svd )





# -------------------


def compute_kmeans():

    rsme_kmeans_users = []
    rsme_kmeans_movies = []
    rsme_kmeans_simmvs = []

    num_clusters_users = 6 # best value so far
    kmeans_users = KMeans(init="k-means++", n_clusters=num_clusters_users, n_init=4, random_state=0).fit(data)
    """
    # save model
    filename = "kmeans_users_" + str(num_clusters_users) + ".sav"
    pickle.dump(kmeans_users, open(filename, 'wb'))
    """
    user_labels = kmeans_users.labels_ # user_labels[user] == cluster that user is part of


    num_clusters_movies = 2 # best value so far
    kmeans_movies = KMeans(init="k-means++", n_clusters=num_clusters_movies, n_init=4, random_state=0).fit(np.transpose(data))
    """
    # save model
    filename = "kmeans_movies_" + str(num_clusters_movies) + ".sav"
    pickle.dump(kmeans_movies, open(filename, 'wb'))
    """
    movie_labels = kmeans_movies.labels_  # movie_labels[movie] == cluster that movie is part of

    cluster_list_users = get_cluster_list(num_clusters_users, user_labels) # cluster_list_users[cluster] == list of users in cluster
    cluster_list_movies = get_cluster_list(num_clusters_movies, movie_labels) # cluster_list_movies[cluster] == list of movies in cluster

    avg_rat_user_cluster, avg_rat_movie_cluster = get_avg_of_clusters(cluster_list_users, cluster_list_movies)
    similar_movies = sim_movie_avg(cluster_list_movies)


    print("predicting...")
    predict_data(1, similar_movies, rsme_kmeans_users, rsme_kmeans_movies, rsme_kmeans_simmvs)

    print("predictions users")
    print(rsme_kmeans_users)
    print("predictions movies")
    print(rsme_kmeans_movies)
    print("predictions simmvs")
    print(rsme_kmeans_simmvs)
        


