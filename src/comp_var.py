
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pathlib
from numpy.core.fromnumeric import var
from tqdm import tqdm

path = pathlib.Path().resolve() # working directory
print(path)
path = pathlib.Path(__file__).parent.resolve() # current file
print(path)


def ps(inp, cmt=""):
   print(inp.shape, cmt)

load_dir = 'save_new/'
save_dir = 'local_comp/'
calc_all = True

#load
filepath = os.path.join(load_dir, '%s.sav' % ('rated_movies_per_user'))
with open(filepath, 'rb') as f:
   num_rats_per_user = pickle.load(f)
filepath = os.path.join(load_dir, '%s.sav' % ('rated_users_per_movie'))
with open(filepath, 'rb') as f:
    num_rats_per_movie = pickle.load(f)
filepath = os.path.join(load_dir, '%s.sav' % ('predictions'))
with open(filepath, 'rb') as f:
    pred = pickle.load(f)


def nested_list_zip(a, b):
   output = []
   iter_len = min(len(a), len(b))+1
   for i in range(iter_len):
      if i == len(a):
         output = output + b[i:]
      elif i == len(b):
         output = output + a[i:]
      else:
         output.append(a[i]+b[i])
   return output
      



ps(num_rats_per_movie)
ps(num_rats_per_user)

def save_obj(obj, name):
   filepath = os.path.join(save_dir, '%s.sav' % name)
   with open(filepath, 'wb') as f:
      pickle.dump(obj, f)

def load_obj(name, load_path=load_dir):
   filepath = os.path.join(load_path, '%s.sav' % name)
   with open(filepath, 'rb') as f:
      obj = pickle.load(f)
   return obj
# movies per num ratings
#users_per_num_rat = [np.where(pred == i_id) for i_id in range(num_rats_per_user.max()+1)]
#movies_per_num_rat = [np.where(pred == i_id) for i_id in range(num_rats_per_movie.max()+1)]

def extract_indices(input):
   max_val = input.max()
   print('max value: ', max_val)
   output = []
   with tqdm(total=((max_val+1)* input.shape[0]) )  as pbar:
      for i in range(max_val+1):
         temp = []
         for idx in range(input.shape[0]):
            if input[idx] == i:
               temp.append(idx)
            pbar.update(1)
         output.append(np.array(temp))
   return output


var_pred = np.var(pred, 0)
var_pred = np.std(pred,0)




u_max = num_rats_per_user.max()
m_max = num_rats_per_movie.max()
calc_all = False
if calc_all:
   aux_users = [ [] for _ in range(num_rats_per_user.max()+1) ]
   aux_movies = [ [] for _ in range(num_rats_per_movie.max()+1) ]
   aux_comb = [ [] for _ in range(num_rats_per_movie.max()+num_rats_per_user.max()*ratio+2) ]
   #idx -> num rat: cont-> list of vars#


   with tqdm(total=(var_pred.shape[0]*var_pred.shape[1]) )  as pbar:
      for i in range(var_pred.shape[0]):
         for j in range(var_pred.shape[1]):
            aux_users[num_rats_per_user[j]].append(var_pred[i][j])
            aux_movies[num_rats_per_movie[i]].append(var_pred[i][j])
            u = num_rats_per_user[j] 
            m = num_rats_per_movie[i] 
            #aux_comb[num_rats_per_user[j]+num_rats_per_movie[i]].append(var_pred[i][j])
            aux_comb[u+m].append(var_pred[i][j])
            pbar.update(1)

def aux_per_row(row_idx, tp_row=False):
   max_len = u_max
   num_rats = num_rats_per_user
   len_var_pred = var_pred.shape[1]
   if tp_row:
      max_len = m_max
      num_rats = num_rats_per_movie
      len_var_pred = var_pred.shape[0]
   aux_list = [ [] for _ in range(max_len+1) ]


   for j in range(len_var_pred):
      if not tp_row:
         aux_list[num_rats[j]].append(var_pred[row_idx][j])
      else:
         aux_list[num_rats[j]].append(var_pred[j][row_idx])
   return aux_list


      

def aux_per_rows(row_idxs, tp_row=False):
   if len(row_idxs) == 0:
      print("input len is 0")
      return
   with tqdm(total=(len(row_idxs)) )  as pbar:
      output = aux_per_row(row_idxs[0], tp_row)
      pbar.update(1)
      for i in row_idxs[1:]:
         t = aux_per_row(i, tp_row)
         assert len(t) == len(output)
         output = nested_list_zip(output, t)
         pbar.update(1)
   return output




def sliding_smoothing(data_array, window=5, mode='mean'):  
   #data_array = np.array(data_array)  
   data_min = min(data_array)
   data_max = max(data_array)
   new_list = []  
   for i in range(len(data_array)):  
      indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, len(data_array)))  
      if mode == 'mean':
         avg = 0  
         for j in indices:  
            avg += data_array[j]  
         avg /= float(len(indices))  
         new_list.append(avg)  
      elif mode == 'max':
         max_val = data_min
         for j in indices:
            if data_array[j] > max_val:
               max_val=data_array[j]
         new_list.append(max_val)  
      elif mode == 'min':
         min_val = data_max
         for j in indices:
            if data_array[j] < min_val:
               min_val=data_array[j]
         new_list.append(min_val)  
   return new_list 


def prep_plot(input):
   x_axis = []
   y_axis = []
   max_y = []
   min_y = []
   for i in range(len(input)):
      if len(input[i]) > 0:
         arr = np.array(input[i])
         x_axis.append(i)
         max_y.append(arr.max())
         min_y.append(arr.min())
         y_axis.append(np.mean(arr))

   return x_axis, y_axis, min_y, max_y

def make_fig(x, y, min_y, max_y, lim, xlabel, file_name, ylabel="standard deviation of predictions",):
   fig, ax = plt.subplots(figsize=(16,9))
   ax.plot(x, y, label="average")
   ax.plot(x, max_y, label="upper bound")
   ax.plot(x, min_y, label="lower bound")

   ax.set_ylim(lim)
   ax.legend(fontsize=20)
   ax.spines["top"].set_visible(False)    
   ax.spines["right"].set_visible(False) 
   plt.yscale('linear')
   plt.xscale('linear')

   plt.ylabel(ylabel, fontsize=26) 
   plt.xlabel(xlabel, fontsize=26) 
   plt.xticks(fontsize=20)  
   plt.yticks(fontsize=20)  
   plt.savefig(file_name, format='png',bbox_inches='tight')   
   plt.close() 





# avg all
if calc_all:
   user_x, user_y, user_min, user_max = prep_plot(aux_users)
   movie_x, movie_y, movie_min, movie_max = prep_plot(aux_movies)
   user_max = sliding_smoothing(user_max,3, mode='mean')
   movie_max = sliding_smoothing(movie_max, mode='mean')
   movie_y = sliding_smoothing(movie_y, 3, 'mean')
   make_fig(user_x,user_y, user_min, user_max, [-0.1, 2], "number of movies rated",'std_predictions_user.png')
   make_fig(movie_x, movie_y, movie_min, movie_max, [-0.1, 2], "number of users rating",'std_predictions_movies.png')

def find_val(input, val):
   for i in range(len(input)):
      if input[i] == val:
         return i
   return -1

def find_vals(input, val):
   output = []
   for i in range(len(input)):
      if input[i] == val:
         output.append(i)
   return output

def find_lowest_non_zero(input, val=0):
   for i in range(val, input.max()):
      t = find_val(input, i)
      if t > -1:
         return t
   return -1

def find_lowest_non_zeros(input, val=0):
   for i in range(val, input.max()):
      t = find_vals(input, i)
      if len(t) != 0:
         return t
   return []

"""
# fixed row
for i in range(0, 100, 10):
   r = find_lowest_non_zero(num_rats_per_movie, i)
   c = find_lowest_non_zero(num_rats_per_user, i)
   rx, ry, rmin, rmax = prep_plot(aux_per_row(r))
   cx, cy, cmin, cmax = prep_plot(aux_per_row(c, True))
   ry = sliding_smoothing(ry, mode='mean')
   rmin = sliding_smoothing(rmin, mode='mean')
   rmax = sliding_smoothing(rmax, mode='mean')
   cy = sliding_smoothing(cy, mode='mean')
   cmin = sliding_smoothing(cmin, mode='mean')
   cmax = sliding_smoothing(cmax, mode='mean')
   make_fig(rx, ry, rmin, rmax, [-0.1, 3], "numbers of movies a user has rated",'std_predictions_fixed_movie_%d.png' % i)
   make_fig(cx, cy, cmin, cmax, [-0.1, 3], "number of ratings a movie has",'std_predictions_fixed_user_%d.png' % i)
"""
# fixed val rows
for i in range(0, 100, 10):
   rs = find_lowest_non_zeros(num_rats_per_movie,i)
   cs = find_lowest_non_zeros(num_rats_per_user,i)

   rsx, rsy, rsmin, rsmax = prep_plot(aux_per_rows(rs))
   csx, csy, csmin, csmax = prep_plot(aux_per_rows(cs, True))
   #rsy = sliding_smoothing(rsy, mode='mean')
   #rsmin = sliding_smoothing(rsmin, mode='mean')
   #rsmax = sliding_smoothing(rsmax, mode='mean')
   #csy = sliding_smoothing(csy, mode='mean')
   #csmin = sliding_smoothing(csmin, mode='mean')
   #csmax = sliding_smoothing(csmax, mode='mean')
   make_fig(rsx, rsy, rsmin, rsmax, [-0.1, 4], "numbers of movies a user has rated",'std_predictions_fixed_movies_%d.png' % i, "standard deviation of predictions\nfor movies with %d user ratings" % num_rats_per_movie[rs[0]] )
   make_fig(csx, csy, csmin, csmax, [-0.1, 4], "number of ratings a movie has",'std_predictions_fixed_users_%d.png' % i,"standard deviation of predictions\nfor users with %d movie ratings" % num_rats_per_user[cs[0]] )

mx, my, mmin, mmax = [], [], [], []
for i in range(0, 60, 20):
   rs = find_lowest_non_zeros(num_rats_per_movie,i)
   rsx, rsy, rsmin, rsmax = prep_plot(aux_per_rows(rs))
   rsy = sliding_smoothing(rsy, mode='mean')
   rsmin = sliding_smoothing(rsmin, mode='mean')
   rsmax = sliding_smoothing(rsmax, mode='mean')
   mx.append(rsx)
   my.append(rsy)
   mmin.append(rsmin)
   mmax.append(rsmax)

fig, ax = plt.subplots(1,3)
ax[0].plot(mx[0], my[0], label="average")
ax[0].plot(mx[0], my[0], label="upper bound")
ax[0].plot(mx[0], my[0], label="lower bound")
ax[0].set_ylim([-0.1,4])
ax[0].spines["top"].set_visible(False)    
ax[0].spines["right"].set_visible(False) 

ax[1].plot(mx[1], my[1], label="average")
ax[1].plot(mx[1], my[1], label="upper bound")
ax[1].plot(mx[1], my[1], label="lower bound")

ax[2].plot(mx[2], my[2], label="average")
ax[2].plot(mx[2], my[2], label="upper bound")
ax[2].plot(mx[2], my[2], label="lower bound")

ax[2].legend(fontsize=20)

plt.yscale('linear')
plt.xscale('linear')

plt.ylabel("standard deviation of predictions for movies\nwith 8, 27, and 41 user ratings", fontsize=26) 
plt.xlabel("numbers of movies a user has rated", fontsize=26) 
plt.xticks(fontsize=20)  
plt.yticks(fontsize=20)  
plt.savefig("std_fixed_movies_subplots.png", format='png',bbox_inches='tight')   
plt.close() 


def clip(input, clip_val):
   for i in range(len(input)):
      if input[i]>clip_val:
         input[i]=clip_val





