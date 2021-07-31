# ## Project 1 - Collaborative Filtering
# 
# ### ETH Computational Intelligence Lab 2021 - Project 1 
# 
# Disclaimer: Most methods mentioned here are briefly presented and not optimized. A solid baseline comparison may explore ways to optimize these. 
# 
# The problem of collaborative filtering concerns providing users with personalized product recommendations. The growth of e-commerce and social media platforms has established the need for recommender systems capable of providing personalized product recommendations. Here we resolve to past user behavior and exploit data dependencies to predict preferences for specific user-item interactions. 
# 
# This problem attracted great interest by the introduction of the [Netflix Prize](https://www.netflixprize.com/) that aimed to improve recommendations of Netflix's own algorithm. Given a list of users-items interactions, the task is to predict a series of ratings for another list of future interactions.

# In our setting, we are dealing with a smaller dataset. 
# 
# We are given a set of $N = 1,176,952$ integer movie ratings, ranging from $1$ to $5$, that are assigned by $m=10,000$ users to $n=1,000$ movies. A rating $r_{ui}$ indicates the preference by user $u$ of item $i$. Let $\mathcal{\Omega} = \{(u,i) : r_{ui} \text{ is known} \}$ be the set of user and movie indices for which the ratings are known. 

import pandas as pd
import numpy as np
import math


# To download the data make sure you have joined the kaggle competition. Then create an api key through kaggle.


number_of_users, number_of_movies = (10000, 1000)

data_pd = pd.read_csv('../../data/data_train.csv')
print(data_pd.head(5))
print()
print('Shape', data_pd.shape)

eval_pd = pd.read_csv('../../data/sampleSubmission.csv')


# The provided data $\mathcal{\Omega}$ are split into two disjoint subsets, namely $\mathcal{\Omega}_{\text{train}}$ and $\mathcal{\Omega}_{\text{test}}$. The former consists of $90\%$ of the data and is used for training the individual models whereas the latter consists of the remaining $10\%$ of the data and is used for learning optimal blending weights. Depending on your method, you may choose to retrain on the whole dataset for the final solution.


from sklearn.model_selection import train_test_split
# Split the dataset into train and test

train_size = 0.9

train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=42)


def extract_users_items_predictions(data_pd):
    users, movies =         [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    predictions = data_pd.Prediction.values
    return users, movies, predictions

train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, train_predictions):
    data[user][movie] = pred
    mask[user][movie] = 1


# To consider:
# * Do unobserved values matter for the method we are using? If yes, is the above initialization the best?
# * Does normalization of the data matter for the method we are using?
# * If yes, should we do the same transformations for the test data?
# 
# Our task is to predict ratings according to specific users-movies combinations. We will quantify the quality of our predictions based on the root mean squared error (RMSE) function between the true and observed ratings. For a given set of observations $ \mathcal{\Omega}$, let
# \begin{equation}
#     \text{RMSE} = \sqrt{\frac{1}{|\mathcal{\Omega}|}\sum_{ (u,i) \in \mathcal{\Omega}} \big(r_{ui} - \hat{r}_{ui}\big)^2}
# \end{equation}
# where $\hat{r}_{ui}$ denotes the estimate of $r_{ui}$.


from sklearn.metrics import mean_squared_error

rmse = lambda x, y: math.sqrt(mean_squared_error(x, y))

test_users, test_movies, test_predictions = extract_users_items_predictions(test_pd)

# test our predictions with the true values
def get_score(predictions, target_values=test_predictions):
    return rmse(predictions, target_values)

def extract_prediction_from_full_matrix(reconstructed_matrix, users=test_users, movies=test_movies):
    # returns predictions for the users-movies combinations specified based on a full m \times n matrix
    assert(len(users) == len(movies)), "users-movies combinations specified should have equal length"
    predictions = np.zeros(len(users))

    for i, (user, movie) in enumerate(zip(users, movies)):
        predictions[i] = reconstructed_matrix[user][movie]

    return predictions


# ### Evaluate Model


eval_users, eval_movies, _ = extract_users_items_predictions(eval_pd)

def evaluate_model(model_name, reconstructed_matrix, unstandardize=False):
    predictions = extract_prediction_from_full_matrix(reconstructed_matrix, eval_users, eval_movies)
#     if unstandardize:
#         predictions = unstandardize_test_predictions(eval_users, predictions, train_user_prediction_mean_mapping, train_user_prediction_std_mapping)
    eval_pd['Prediction'] = predictions
    eval_pd.to_csv(f'{model_name}.csv', index=False)


# # Methods and Approaches

# ## General Average


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))

reconstructed_matrix = data

predictions = extract_prediction_from_full_matrix(reconstructed_matrix)

print("RMSE for General Average: {:.4f}".format(get_score(predictions)))

evaluate_model('gen_avg', reconstructed_matrix)


# ## User Average


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

def calculate_user_means(train_users, train_predictions):
    train_user_predictions_mapping = {}
    for train_user, train_prediction in zip(train_users, train_predictions):
        if train_user in train_user_predictions_mapping.keys():
            train_user_predictions_mapping[train_user].append(train_prediction)
        else:
            train_user_predictions_mapping[train_user] = [train_prediction]
    train_user_prediction_mean_mapping = {}
    for key, value in train_user_predictions_mapping.items():
        train_user_prediction_mean_mapping[key] = np.mean(value)
    return train_user_prediction_mean_mapping

data = np.full((number_of_users, number_of_movies), 0)

train_user_prediction_mean_mapping = calculate_user_means(train_users, train_predictions)

for user, movie in zip(test_users.tolist(), test_movies.tolist()):
    data[user][movie] = train_user_prediction_mean_mapping[user]

reconstructed_matrix = data

predictions = extract_prediction_from_full_matrix(reconstructed_matrix)

print("RMSE for User Average: {:.4f}".format(get_score(predictions)))

evaluate_model('user_avg', reconstructed_matrix)


# ## Movie Average


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

def calculate_movie_means(train_movies, train_predictions):
    train_movie_predictions_mapping = {}
    for train_movie, train_prediction in zip(train_movies, train_predictions):
        if train_movie in train_movie_predictions_mapping.keys():
            train_movie_predictions_mapping[train_movie].append(train_prediction)
        else:
            train_movie_predictions_mapping[train_movie] = [train_prediction]
    train_movie_prediction_mean_mapping = {}
    for key, value in train_movie_predictions_mapping.items():
        train_movie_prediction_mean_mapping[key] = np.mean(value)
    return train_movie_prediction_mean_mapping

# also create full matrix of observed values
data = np.full((number_of_users, number_of_movies), 0)

train_movie_prediction_mean_mapping = calculate_movie_means(train_movies, train_predictions)

for user, movie in zip(test_users.tolist(), test_movies.tolist()):
    data[user][movie] = train_user_prediction_mean_mapping[movie]

reconstructed_matrix = data

predictions = extract_prediction_from_full_matrix(reconstructed_matrix)

print("RMSE for Movie Average: {:.4f}".format(get_score(predictions)))

evaluate_model('movie_avg', reconstructed_matrix)


# ## Singular Value Decomposition (SVD)
# 
# Assuming that column entries are not random, we attempt to fill the missing entries by capturing some of the most significant components of the underlying data. Assume that a latent factor model associates each user $u$ with a set of user factors $ p_u \in \mathbb{R}^k$ and each item $i$ with a set of item factors $ q_i \in \mathbb{R}^k$. In the case of movies, some of these factors could correspond to movie genres such as comedy, drama, action etc. For each item $i$, the elements of $ q_i$ quantify the extent to which the item possesses these factors. Similarly, for each user $u$, the elements of $ p_u$ measure the level of interest that the user has to each of these factors. In this framework, user-item interactions are modeled by inner products in the latent space, leading to the following prediction rule 
# \begin{equation}
# \hat{r}_{ui} =  p_u^T q_i.
# \end{equation}
# 
# Singular Value Decomposition (SVD) [1] is a widely used technique for matrix factorization. Any matrix $ M \in \mathbb{R}^{m \times n}$ can be decomposed into $A = U \Sigma V^T$, where $ U \in \mathbb{R}^{m\times m}$, $ \Sigma \in \mathbb{R}^{m \times n}$ and $ V \in \mathbb{R}^{n \times n}$. Matrices $ U$ and $ V$ are orthogonal, whereas $ \Sigma$ has $ rank(A)$ positive entries on the main diagonal sorted in decreasing order of value. 
# 
# We apply the SVD on the imputed user-item matrix to decompose it into $A =  U  \Sigma  V^T$. We may assume that a list of $k$ distinguishes users' interests and movies' characteristics. This motivates us to approximate $A$ by another matrix of low rank. The Eckart-Young theorem [2] states that the optimal (in terms of the Frobenius norm objective) rank $k$ approximation of the matrix $A$ is given by $A_k =  U_k  \Sigma_k  V^T_k$, where $ U_k \in \mathbb{R}^{m \times k}$, $ \Sigma_k \in \mathbb{R}^{k \times k}$ and $ V_k \in \mathbb{R}^{n \times k}$. $ U_k$ and $ V_k$ correspond to the first $k$ columns of $ U$ and $ V$ respectively and $ \Sigma_k$ to the $k \times k$ sub-matrix of $ \Sigma$ containing the $k$ largest singular values. 
# 
# ----------------
# [1] Klema, Virginia, and Alan Laub. "The singular value decomposition: Its computation and some applications." IEEE Transactions on automatic control 25.2 (1980): 164-176.
# 
# [2] Eckart, Carl, and Gale Young. "The approximation of one matrix by another of lower rank." Psychometrika 1.3 (1936): 211-218.

# How many singular values should we keep? Try them all!
# This is why we first use a train-validation split.

# ## Vanilla SVD


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

data = np.full((number_of_users, number_of_movies), np.mean(train_pd.Prediction.values))
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, train_predictions):
    data[user][movie] = pred
    mask[user][movie] = 1

k_singular_values = 10
number_of_singular_values = min(number_of_users, number_of_movies)

assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"

U, s, Vt = np.linalg.svd(data, full_matrices=False)

S = np.zeros((number_of_movies, number_of_movies))
S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

reconstructed_matrix = U.dot(S).dot(Vt)
    
predictions = extract_prediction_from_full_matrix(reconstructed_matrix)

print("RMSE using SVD is: {:.4f}".format(get_score(predictions)))

evaluate_model('svd', reconstructed_matrix)


# ## SVD with user-based standardization


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

def standardize_train_predictions(train_users, train_predictions):
    train_user_predictions_mapping = {}
    for train_user, train_prediction in zip(train_users, train_predictions):
        if train_user in train_user_predictions_mapping.keys():
            train_user_predictions_mapping[train_user].append(train_prediction)
        else:
            train_user_predictions_mapping[train_user] = [train_prediction]

    train_user_prediction_mean_mapping = {}
    train_user_prediction_std_mapping = {}
    for key, value in train_user_predictions_mapping.items():
        train_user_prediction_mean_mapping[key] = np.mean(value)
        train_user_prediction_std_mapping[key] = np.std(value)

    df = pd.DataFrame.from_dict({"train_user": train_users, "train_predictions": train_predictions})
    df["train_predictions"] = df.apply(lambda x: (x["train_predictions"] - train_user_prediction_mean_mapping[x["train_user"]]) / train_user_prediction_std_mapping[x["train_user"]], axis=1)
    return df["train_predictions"].values, train_user_prediction_mean_mapping, train_user_prediction_std_mapping

def unstandardize_test_predictions(test_users, test_predictions, train_user_prediction_mean_mapping, train_user_prediction_std_mapping):
    df = pd.DataFrame.from_dict({"test_user": test_users, "test_predictions": test_predictions})
    df["test_predictions"] = df.apply(lambda x: (x["test_predictions"] * train_user_prediction_std_mapping[x["test_user"]]) + train_user_prediction_mean_mapping[x["test_user"]], axis=1)
    return df["test_predictions"].values

standardized_train_predictions, train_user_prediction_mean_mapping, train_user_prediction_std_mapping = standardize_train_predictions(train_users, train_predictions)

unstandardized_train_predictions = unstandardize_test_predictions(train_users, standardized_train_predictions, train_user_prediction_mean_mapping, train_user_prediction_std_mapping)

assert np.allclose(unstandardized_train_predictions, train_predictions), "Unstandardized train predictions are different than original train predictions"

data = np.full((number_of_users, number_of_movies), 0)
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, standardized_train_predictions):
    data[user][movie] = pred
    mask[user][movie] = 1

k_singular_values = 10
number_of_singular_values = min(number_of_users, number_of_movies)

assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"

U, s, Vt = np.linalg.svd(data, full_matrices=False)

S = np.zeros((number_of_movies, number_of_movies))
S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

reconstructed_matrix = U.dot(S).dot(Vt)

standardized_test_predictions = extract_prediction_from_full_matrix(reconstructed_matrix)
unstandardized_test_predictions = unstandardize_test_predictions(test_users, standardized_test_predictions, train_user_prediction_mean_mapping, train_user_prediction_std_mapping)

print("RMSE for SVD with user-based standardization: {:.4f}".format(get_score(unstandardized_test_predictions)))


evaluate_model('svd_user', reconstructed_matrix, unstandardize=False)


# ## SVD with movie based standardization


train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)

def standardize_train_predictions(train_movies, train_predictions):
    train_movie_predictions_mapping = {}
    for train_movie, train_prediction in zip(train_movies, train_predictions):
        if train_movie in train_movie_predictions_mapping.keys():
            train_movie_predictions_mapping[train_movie].append(train_prediction)
        else:
            train_movie_predictions_mapping[train_movie] = [train_prediction]

    train_movie_prediction_mean_mapping = {}
    train_movie_prediction_std_mapping = {}
    for key, value in train_movie_predictions_mapping.items():
        train_movie_prediction_mean_mapping[key] = np.mean(value)
        train_movie_prediction_std_mapping[key] = np.std(value)

    df = pd.DataFrame.from_dict({"train_movie": train_movies, "train_predictions": train_predictions})
    df["train_predictions"] = df.apply(lambda x: (x["train_predictions"] - train_movie_prediction_mean_mapping[x["train_movie"]]) / train_movie_prediction_std_mapping[x["train_movie"]], axis=1)
    return df["train_predictions"].values, train_movie_prediction_mean_mapping, train_movie_prediction_std_mapping

def unstandardize_test_predictions(test_movies, test_predictions, train_movie_prediction_mean_mapping, train_movie_prediction_std_mapping):
    df = pd.DataFrame.from_dict({"test_movie": test_movies, "test_predictions": test_predictions})
    df["test_predictions"] = df.apply(lambda x: (x["test_predictions"] * train_movie_prediction_std_mapping[x["test_movie"]]) + train_movie_prediction_mean_mapping[x["test_movie"]], axis=1)
    return df["test_predictions"].values

standardized_train_predictions, train_movie_prediction_mean_mapping, train_movie_prediction_std_mapping = standardize_train_predictions(train_movies, train_predictions)

unstandardized_train_predictions = unstandardize_test_predictions(train_movies, standardized_train_predictions, train_movie_prediction_mean_mapping, train_movie_prediction_std_mapping)

assert np.allclose(unstandardized_train_predictions, train_predictions), "Unstandardized train predictions are different than original train predictions"

data = np.full((number_of_users, number_of_movies), 0)
mask = np.zeros((number_of_users, number_of_movies)) # 0 -> unobserved value, 1->observed value

for user, movie, pred in zip(train_users, train_movies, standardized_train_predictions):
    data[user][movie] = pred
    mask[user][movie] = 1

k_singular_values = 10
number_of_singular_values = min(number_of_users, number_of_movies)

assert(k_singular_values <= number_of_singular_values), "choose correct number of singular values"

U, s, Vt = np.linalg.svd(data, full_matrices=False)

S = np.zeros((number_of_movies, number_of_movies))
S[:k_singular_values, :k_singular_values] = np.diag(s[:k_singular_values])

reconstructed_matrix = U.dot(S).dot(Vt)

standardized_test_predictions = extract_prediction_from_full_matrix(reconstructed_matrix)
unstandardized_test_predictions = unstandardize_test_predictions(test_movies, standardized_test_predictions, train_movie_prediction_mean_mapping, train_movie_prediction_std_mapping)

print("RMSE for SVD with movie based standardization: {:.4f}".format(get_score(unstandardized_test_predictions)))

evaluate_model('svd_movie', reconstructed_matrix)

