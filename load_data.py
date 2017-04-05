import gzip
import numpy as np
import pickle
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split


def to_one_hot(yy):
    y = np.zeros((yy.shape[0], 10), dtype=np.int32)
    y[np.arange(yy.shape[0]), yy] = 1
    return y


def load_data():
    dataset = 'C:/Users/phulo/Downloads/mnist.pkl.gz'

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except Exception:
            train_set, valid_set, test_set = pickle.load(f)
    return (train_set[0],to_one_hot(train_set[1])), (valid_set[0],to_one_hot(valid_set[1])),(test_set[0],to_one_hot(test_set[1])),


def load_data_bis():
    """
    Loads Mnist data from the local folder or downloads it if not available.
    mldata.org is currently down so this function is deprecated
    """
    mnist = fetch_mldata('MNIST original', data_home="MNIST")
    X = mnist.data.astype(np.float32)
    Y = mnist.target.astype(np.int32)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=10000, random_state=78)
    Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size=10000, random_state=90)

    return {
        "X_train": Xt,
        "Y_train": to_one_hot(Yt),
        "X_valid": Xv,
        "Y_valid": to_one_hot(Yv),
        "X_test":  X_test,
        "Y_test":  to_one_hot(Y_test),
    }


if __name__ == "__main__":
    X = load_data_bis()
    print(X)
