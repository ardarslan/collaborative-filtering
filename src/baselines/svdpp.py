bayes_by_backprop = False
random_state = 42
import os
import numpy as np; np.random.seed(random_state)
import torch; torch.manual_seed(random_state)
import random; random.seed(random_state)
import pandas as pd
from sklearn.model_selection import train_test_split

data_pd = pd.read_csv('../../data/data_train.csv')

# Parameters
num_epochs = 100
show_validation_score_every_epochs = 1
embedding_size = 200
learning_rate = 7e-4
l1_reg = 1e-5
l2_reg = 0
mean_init = 0.2
std_init = 0.001
train_size = 0.9
batch_size = 64
num_workers = 5 # data_loader

if bayes_by_backprop:
    logsigma_constant_init = -4.6 # -3.9 # -6.90725523732
    prior_mu = 0.0 # 0.2
    prior_sigma = 0.01
    mu_mean_init = 0.0
    mu_std_init = 0.01
    KL_coeff = 1/600000

train_pd, test_pd = train_test_split(data_pd, train_size=train_size, random_state=random_state)

def extract_users_items_labels(data_pd):
    users, movies =         [np.squeeze(arr) for arr in np.split(data_pd.Id.str.extract('r(\d+)_c(\d+)').values.astype(int) - 1, 2, axis=-1)]
    labels = data_pd.Prediction.values
    return users, movies, labels

train_users, train_movies, train_labels = extract_users_items_labels(train_pd)
test_users, test_movies, test_labels = extract_users_items_labels(test_pd)

movies_rated_by_user_u = {}
for train_user, train_movie in zip(train_users, train_movies):
    if train_user in movies_rated_by_user_u.keys():
        movies_rated_by_user_u[train_user].append(train_movie + 1)
    else:
        movies_rated_by_user_u[train_user] = [train_movie + 1]
largest_number_of_ratings_per_user = max(len(movies) for user, movies in movies_rated_by_user_u.items())

users_who_rated_movie_i = {}
for train_user, train_movie in zip(train_users, train_movies):
    if train_movie in users_who_rated_movie_i.keys():
        users_who_rated_movie_i[train_movie].append(train_user + 1)
    else:
        users_who_rated_movie_i[train_movie] = [train_user + 1]
largest_number_of_ratings_per_movie = max(len(users) for movie, users in users_who_rated_movie_i.items())

sqrt_of_number_of_movies_rated_by_user_u = dict((user, np.sqrt(len(movies))) for user, movies in movies_rated_by_user_u.items())
sqrt_of_number_of_users_who_rated_movie_i = dict((movie, np.sqrt(len(users))) for movie, users in users_who_rated_movie_i.items())

is_known_user = dict((user, 0.0) for user in test_users)
for train_user in train_users:
    is_known_user[train_user] = 1.0

is_known_movie = dict((movie, 0.0) for movie in test_movies)
for train_movie in train_movies:
    is_known_movie[train_movie] = 1.0



from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

class MyDataset(Dataset):
    def __init__(self,
                 users,
                 movies,
                 labels,
                 movies_rated_by_user_u,
                 users_who_rated_movie_i,
                 sqrt_of_number_of_movies_rated_by_user_u,
                 sqrt_of_number_of_users_who_rated_movie_i,
                 is_known_user,
                 is_known_movie):
        self.users = users
        self.movies = movies
        self.labels = labels
        self.movies_rated_by_user_u = movies_rated_by_user_u
        self.users_who_rated_movie_i = users_who_rated_movie_i
        self.sqrt_of_number_of_movies_rated_by_user_u = sqrt_of_number_of_movies_rated_by_user_u
        self.sqrt_of_number_of_users_who_rated_movie_i = sqrt_of_number_of_users_who_rated_movie_i
        self.is_known_user = is_known_user
        self.is_known_movie = is_known_movie

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        current_user = self.users[idx]
        current_movie = self.movies[idx]
        data_out = {
            "user": np.array([self.users[idx]]),
            "movie": np.array([self.movies[idx]]),
            "movies_rated_by_this_user": np.array(self.movies_rated_by_user_u[current_user]),
            "users_who_rated_this_movie": np.array(self.users_who_rated_movie_i[current_movie]),
            "sqrt_of_number_of_movies_rated_by_this_user": np.array([self.sqrt_of_number_of_movies_rated_by_user_u[current_user]]),
            "sqrt_of_number_of_users_who_rated_this_movie": np.array([self.sqrt_of_number_of_users_who_rated_movie_i[current_movie]]),
            "is_known_user": np.array([self.is_known_user[current_user]]),
            "is_known_movie": np.array([self.is_known_movie[current_movie]]),
        }
        if self.labels is not None:
            data_out["label"] = self.labels[idx]

        data_out.update(
            {
                key: val.astype(np.float32)
                for key, val in data_out.items()
                if isinstance(val, np.ndarray) and val.dtype == np.float64
            }
        )
        return data_out
    
    @staticmethod
    def collate_fn(data):
        batch = []
        for sample in data:
            sample["movies_rated_by_this_user"].resize(largest_number_of_ratings_per_user)
            sample["users_who_rated_this_movie"].resize(largest_number_of_ratings_per_movie)
            batch.append(sample)
        batch = default_collate(batch)
        return batch

train_dataset = MyDataset(
    users=train_users,
    movies=train_movies,
    labels=train_labels,
    movies_rated_by_user_u=movies_rated_by_user_u,
    users_who_rated_movie_i=users_who_rated_movie_i,
    sqrt_of_number_of_movies_rated_by_user_u=sqrt_of_number_of_movies_rated_by_user_u,
    sqrt_of_number_of_users_who_rated_movie_i=sqrt_of_number_of_users_who_rated_movie_i,
    is_known_user=is_known_user,
    is_known_movie=is_known_movie
)

test_dataset = MyDataset(
    users=test_users,
    movies=test_movies,
    labels=test_labels,
    movies_rated_by_user_u=movies_rated_by_user_u,
    users_who_rated_movie_i=users_who_rated_movie_i,
    sqrt_of_number_of_movies_rated_by_user_u=sqrt_of_number_of_movies_rated_by_user_u,
    sqrt_of_number_of_users_who_rated_movie_i=sqrt_of_number_of_users_who_rated_movie_i,
    is_known_user=is_known_user,
    is_known_movie=is_known_movie
)

train_data_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=train_dataset.collate_fn,
    num_workers=num_workers)

test_data_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
    num_workers=num_workers)

def data_2_device(data, device):
    for key in data.keys():
        if torch.is_tensor(data[key]):
            data[key] = data[key].to(device=device)



random_state = 42
import numpy as np
np.random.seed(random_state)
import torch
torch.manual_seed(random_state)
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

number_of_users, number_of_movies = (10000, 1000)

class SVDPP(nn.Module):
    def __init__(self, number_of_users, number_of_movies, embedding_size, global_mean, mean_init, std_init):
        super().__init__()
        self.global_mean = global_mean
        self.Bu = nn.Embedding(number_of_users, 1)
        nn.init.normal_(self.Bu.weight, mean=mean_init, std=std_init)
        self.Bi = nn.Embedding(number_of_movies, 1)
        nn.init.normal_(self.Bu.weight, mean=mean_init, std=std_init)
        self.P = nn.Embedding(number_of_users, embedding_size)
        nn.init.normal_(self.P.weight, mean=mean_init, std=std_init)
        self.Q = nn.Embedding(number_of_movies, embedding_size)
        nn.init.normal_(self.Q.weight, mean=mean_init, std=std_init)
        self.Y = nn.Embedding(number_of_movies + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU. (To pad movies_rated_by_this_user with zeros.)
        nn.init.normal_(self.Y.weight, mean=mean_init, std=std_init)
        # self.Z = nn.Embedding(number_of_users + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU. (To pad movies_rated_by_this_user with zeros.)
        # nn.init.normal_(self.Z.weight, mean=mean_init, std=std_init)

    def forward(self, data):
        users, movies, movies_rated_by_this_user, users_who_rated_this_movie, sqrt_of_number_of_movies_rated_by_this_user, sqrt_of_number_of_users_who_rated_this_movie, is_known_user, is_known_movie = torch.squeeze(data["user"]), torch.squeeze(data["movie"]), data["movies_rated_by_this_user"], data["users_who_rated_this_movie"], data["sqrt_of_number_of_movies_rated_by_this_user"], data["sqrt_of_number_of_users_who_rated_this_movie"], data["is_known_user"], data["is_known_movie"]
        gm = self.global_mean

        bu = self.Bu(users)
        if not self.training:
            bu = is_known_user * bu

        bi = self.Bi(movies)
        if not self.training:
            bi = is_known_movie * bi

        p = self.P(users)
        if not self.training:
            p = is_known_user * p

        q = self.Q(movies)
        if not self.training:
            q = is_known_movie * q

        y = self.Y(movies_rated_by_this_user).sum(dim=1).div(sqrt_of_number_of_movies_rated_by_this_user)
        if not self.training:
            y = is_known_user * y
            

        result = q.mul(p+y).sum(dim=1) + torch.squeeze(bi) + torch.squeeze(bu) + gm
        
        return result
    

class SVDPP_Bayes_by_Backprop(nn.Module):
    def __init__(self,
                 number_of_users,
                 number_of_movies,
                 embedding_size,
                 global_mean,
                 prior_mu,
                 prior_sigma,
                 mu_mean_init,
                 mu_std_init,
                 logsigma_constant_init):
        super().__init__()
        self.embedding_size = embedding_size
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.global_mean = global_mean
        
        self.Bu_mu = nn.Embedding(number_of_users, 1)
        nn.init.normal_(self.Bu_mu.weight, mean=mu_mean_init, std=mu_std_init)
        self.Bu_logsigma = nn.Embedding(number_of_users, 1)
        nn.init.constant_(self.Bu_logsigma.weight, logsigma_constant_init)
        
        self.Bi_mu = nn.Embedding(number_of_movies, 1)
        nn.init.normal_(self.Bi_mu.weight, mean=mu_mean_init, std=mu_std_init)
        self.Bi_logsigma = nn.Embedding(number_of_movies, 1)
        nn.init.constant_(self.Bi_logsigma.weight, logsigma_constant_init)
        
        self.P_mu = nn.Embedding(number_of_users, embedding_size)
        nn.init.normal_(self.P_mu.weight, mean=mu_mean_init, std=mu_std_init)
        self.P_logsigma = nn.Embedding(number_of_users, embedding_size)
        nn.init.constant_(self.P_logsigma.weight, logsigma_constant_init)
        
        self.Q_mu = nn.Embedding(number_of_movies, embedding_size)
        nn.init.normal_(self.Q_mu.weight, mean=mu_mean_init, std=mu_std_init)
        self.Q_logsigma = nn.Embedding(number_of_movies, embedding_size)
        nn.init.constant_(self.Q_logsigma.weight, logsigma_constant_init)
        
        self.Y_mu = nn.Embedding(number_of_movies + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU.
        nn.init.normal_(self.Y_mu.weight, mean=mu_mean_init, std=mu_std_init)
        self.Y_logsigma = nn.Embedding(number_of_movies + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU.
        nn.init.constant_(self.Y_logsigma.weight, logsigma_constant_init)
        
        # self.Z_mu = nn.Embedding(number_of_users + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU.
        # nn.init.normal_(self.Z_mu.weight, mean=mu_mean_init, std=mu_std_init)
        # self.Z_logsigma = nn.Embedding(number_of_users + 1, embedding_size, padding_idx=0) # Made this 1-indexed to save memory in GPU.
        # nn.init.constant_(self.Z_logsigma.weight, logsigma_constant_init)
        

    def forward(self, data):
        users, movies, movies_rated_by_this_user, users_who_rated_this_movie, sqrt_of_number_of_movies_rated_by_this_user, sqrt_of_number_of_users_who_rated_this_movie, is_known_user, is_known_movie = torch.squeeze(data["user"]), torch.squeeze(data["movie"]), data["movies_rated_by_this_user"], data["users_who_rated_this_movie"], data["sqrt_of_number_of_movies_rated_by_this_user"], data["sqrt_of_number_of_users_who_rated_this_movie"], data["is_known_user"], data["is_known_movie"]
        gm = self.global_mean

        bu_mu = self.Bu_mu(users)
        bu_sigma = F.softplus(self.Bu_logsigma(users))
        bu = bu_mu + bu_sigma * torch.normal(mean=torch.zeros_like(bu_mu), std=torch.ones_like(bu_mu))
        if not self.training:
            bu = is_known_user * bu

        bi_mu = self.Bi_mu(movies)
        bi_sigma = F.softplus(self.Bi_logsigma(movies))
        bi = bi_mu + bi_sigma * torch.normal(mean=torch.zeros_like(bi_mu), std=torch.ones_like(bi_mu))
        if not self.training:
            bi = is_known_movie * bi

        p_mu = self.P_mu(users)
        p_sigma = F.softplus(self.P_logsigma(users))
        p = p_mu + p_sigma * torch.normal(mean=torch.zeros_like(p_mu), std=torch.ones_like(p_mu))
        if not self.training:
            p = is_known_user * p

        q_mu = self.Q_mu(movies)
        q_sigma = F.softplus(self.Q_logsigma(movies))
        q = q_mu + q_sigma * torch.normal(mean=torch.zeros_like(q_mu), std=torch.ones_like(q_mu))
        if not self.training:
            q = is_known_movie * q

        y_mu = self.Y_mu(movies_rated_by_this_user)
        y_sigma = F.softplus(self.Y_logsigma(movies_rated_by_this_user))
        y = y_mu + y_sigma * torch.normal(mean=torch.zeros_like(y_mu), std=torch.ones_like(y_mu))
        y = y.sum(dim=1).div(sqrt_of_number_of_movies_rated_by_this_user)
        if not self.training:
            y = is_known_user * y

        result = q.mul(p+y).sum(dim=1) + torch.squeeze(bi) + torch.squeeze(bu) + gm
        
        return result
    
    def kl_divergence(self):
        '''
        Computes the KL divergence between the priors and posteriors of all embeddings.
        '''
        kl_loss = self._kl_divergence(self.Bu_mu.weight, self.Bu_logsigma.weight)
        kl_loss += self._kl_divergence(self.Bi_mu.weight, self.Bi_logsigma.weight)
        kl_loss += self._kl_divergence(self.P_mu.weight, self.P_logsigma.weight)
        kl_loss += self._kl_divergence(self.Q_mu.weight, self.Q_logsigma.weight)
        kl_loss += self._kl_divergence(self.Y_mu.weight, self.Y_logsigma.weight)
        return kl_loss

    def _kl_divergence(self, mu, logsigma):
        '''
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        '''
        sigma = F.softplus(logsigma)
        params = mu + sigma * torch.normal(mean=torch.zeros_like(mu), std=torch.ones_like(mu))
        
        p_prior_dist = torch.distributions.normal.Normal(self.prior_mu, self.prior_sigma)
        p_prior_log_prob = p_prior_dist.log_prob(params)
        
        q_posterior_dist = torch.distributions.normal.Normal(mu, sigma)
        q_posterior_log_prob = q_posterior_dist.log_prob(params)
        
        kl = torch.sum(q_posterior_log_prob - p_prior_log_prob)

        return kl


rmse = lambda x, y: np.sqrt(mean_squared_error(x, y))

def mse_loss(predictions, labels):
    return torch.mean((predictions - labels) ** 2)

global_mean = np.mean(train_labels)

if bayes_by_backprop:
    model = SVDPP_Bayes_by_Backprop(number_of_users,
                                    number_of_movies,
                                    embedding_size,
                                    global_mean,
                                    prior_mu,
                                    prior_sigma,
                                    mu_mean_init,
                                    mu_std_init,
                                    logsigma_constant_init).float().to(device)
else:
    model = SVDPP(number_of_users,
                  number_of_movies,
                  embedding_size,
                  global_mean,
                  mean_init,
                  std_init).float().to(device)

optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=l2_reg)

train_rmse_values = []
test_rmse_values = []
step = 0
with tqdm(total=len(train_data_loader) * num_epochs) as pbar:
    for epoch in range(num_epochs):
        for batch in train_data_loader:
            optimizer.zero_grad()
            
            data_2_device(batch, device)

            predictions_batch = model(batch)
            labels_batch = batch["label"]

            loss = mse_loss(predictions_batch, labels_batch.float())
            
            if bayes_by_backprop:
                loss += (KL_coeff * model.kl_divergence() / batch_size)
            else:
                for embedding_layer in [model.Bu, model.Bi, model.P, model.Q, model.Y]:
                    loss += l1_reg * torch.norm(embedding_layer.weight, 1)

            loss.backward()

            optimizer.step()

            pbar.update(1)

            step += 1
        
        
        if epoch % show_validation_score_every_epochs == 0:
            model.eval()

            with torch.no_grad():
                all_train_predictions = []
                all_train_labels = []
                for batch in train_data_loader:
                    data_2_device(batch, device)
                    predictions_batch = model(batch)
                    labels_batch = batch["label"]
                    all_train_predictions.extend(predictions_batch.detach().cpu().numpy().ravel().tolist())
                    all_train_labels.extend(labels_batch.cpu().numpy().ravel().tolist())

                all_test_predictions = []
                all_test_labels = []
                for batch in test_data_loader:
                    data_2_device(batch, device)
                    predictions_batch = model(batch)
                    labels_batch = batch["label"]
                    all_test_predictions.extend(predictions_batch.detach().cpu().numpy().ravel().tolist())
                    all_test_labels.extend(labels_batch.cpu().numpy().ravel().tolist())

            all_train_labels = np.array(all_train_labels)
            all_test_labels = np.array(all_test_labels)
            all_train_predictions = np.clip(np.array(all_train_predictions), 1, 5)
            all_test_predictions = np.clip(np.array(all_test_predictions), 1, 5)
            train_rmse = rmse(all_train_labels, all_train_predictions)
            test_rmse = rmse(all_test_labels, all_test_predictions)
            print('Epoch: {:3d}, Train RMSE: {:.4f}, Test RMSE: {:.4f}'.format(epoch, train_rmse, test_rmse))
            train_rmse_values.append(train_rmse)
            test_rmse_values.append(test_rmse)

            model.train()



model.eval()

eval_pd = pd.read_csv('./data/sampleSubmission.csv')

eval_users, eval_movies, eval_labels = extract_users_items_labels(eval_pd)

eval_dataset = MyDataset(
    users=eval_users,
    movies=eval_movies,
    labels=eval_labels,
    movies_rated_by_user_u=movies_rated_by_user_u,
    users_who_rated_movie_i=users_who_rated_movie_i,
    sqrt_of_number_of_movies_rated_by_user_u=sqrt_of_number_of_movies_rated_by_user_u,
    sqrt_of_number_of_users_who_rated_movie_i=sqrt_of_number_of_users_who_rated_movie_i,
    is_known_user=is_known_user,
    is_known_movie=is_known_movie
)

eval_data_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=eval_dataset.collate_fn,
    num_workers=num_workers)

all_eval_predictions = []
for batch in eval_data_loader:
    data_2_device(batch, device)
    predictions_batch = model(batch)
    all_eval_predictions.extend(predictions_batch.detach().cpu().numpy().ravel().tolist())
    

eval_pd['Prediction'] = all_eval_predictions
eval_pd.to_csv(f'svdpp_output.csv', index=False)
