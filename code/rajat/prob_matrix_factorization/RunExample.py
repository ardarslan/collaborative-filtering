import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat, load_cf_data
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    file_path = "data/ml-100k/u.data"
    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 200, "num_batches": 100,
                    "batch_size": 1000})
    # ratings_old = load_rating_data(file_path)
    ratings = load_cf_data()
    # ratings = ratings_old

    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.2)  # spilt_rating_dat(ratings)
    pmf.fit(train, test)

    # Check performance by plotting train and test errors
    plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data', markersize=1)
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data', markersize=1)
    plt.title('The MovieLens Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('learning_curve_200_epochs.png')
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
