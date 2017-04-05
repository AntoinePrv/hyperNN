import gzip
import numpy as np
import pickle

def load_data():
    def to_one_hot(yy):
        y = np.zeros((yy.shape[0], 10))
        y[np.arange(yy.shape[0]), yy] = 1
        return y
    #############
    # LOAD DATA #
    #############
    dataset = 'C:/Users/phulo/Downloads/mnist.pkl.gz'

    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    return (train_set[0],to_one_hot(train_set[1])), (valid_set[0],to_one_hot(valid_set[1])),(test_set[0],to_one_hot(test_set[1])),
