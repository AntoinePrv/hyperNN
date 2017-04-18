# import sys
# sys.path.append(".")

from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model
import sklearn.model_selection as mds
import numpy as np
import problem.load_data as load_data
import problem.train_MINST as train_MINST
import matplotlib.pyplot as plt
import hyperLearn.sample as sp

############################################
## function to learn the RSM of the model ##
############################################
def learn_RSM(x, y):
    # x_train, x_test, y_train, y_test = mds.train_test_split(x, y, test_size=0.10, random_state=0)
    # x_train, x_valid, y_train, y_valid = mds.train_test_split(x, y, test_size=0.10, random_state=0)
    # print('RSMdata',x_train,y_train)
    x_train = x
    y_train = y
    ## due to the size of the sets we use all data for training and validation, the RSM may overfit
    x_valid = x
    y_valid = y

    dim = x.shape[1]
    input = Input(shape=(dim,))
    # the network architecture : 2 hidden layers as indicated in the article
    network = Dense(20 * dim, activation="relu")(input)
    network = Dense(20 * dim, activation="relu")(network)
    network = Dense(1, activation="linear")(network)

    # print(x_train,y_train)
    model = Model(input=input, output=network)
    opt = keras.optimizers.SGD(lr=0.1)
    model.compile(optimizer=opt, loss="mse")
    loss = model.fit(x_train, y_train,
                     nb_epoch=100,
                     batch_size=20,
                     verbose=0,
                     shuffle=True,
                     validation_data=(x_valid, y_valid))
    return model.evaluate(x_valid, y_valid, verbose=0), model

## function to add a value to the training set for the RSM
def add_train(old, new):
    to_train = new[0]
    if old[0] == None:
        return np.matrix(to_train), [new[1]]
    else:
        # print(old[0].shape,np.matrix(to_train).shape)
        return np.append(old[0], np.matrix(to_train), axis=0), np.append(old[1], [new[1]], axis=0)

## main function
def learn_hyperparam():
    data = load_data.load_data()
    x = None
    y = None
    # initialisation : 5 random training
    for i in range(5):
        s = sp.sample()
        print(s.get_MNIST())
        res = train_MINST.train_model_s(s.get_MNIST(), data)
        (x, y) = add_train((x, y), (s.get_RSM(), res[0]))
    (acc, RSM_model) = learn_RSM(x, y)

    # initialisation of loops variables
    hist_acc = []
    hist_res = []
    macc = []
    alpha = 0.001
    max_acc = max(y)
    best_model = []
    k = 0
    # first sample is randomly chosen
    s = sp.sample()
    best_conf = s
    while k < 200:
        print('n° test', k)
        # with a little propability we come bach to the best configuration found yet
        if np.random.uniform(0, 1) < alpha * 0.001:
            s = best_conf
        # gaussion sampling of the next solution
        s = s.gaussian_samp()
        # prediction of its performance
        p = RSM_model.predict(np.matrix(s.get_RSM()))
        print('predict_loss ', p, 'acc', max_acc)
        # test conditions :
        if p > max_acc:
            r = np.random.uniform(0, 1)
            if r > alpha:
                RSM_model, best_conf, best_model, k, max_acc, x, y = test_mod(RSM_model, best_conf, best_model, data,
                                                                              hist_acc, hist_res, k, macc, max_acc, s,
                                                                              x, y)
        else:
            r = np.random.uniform(0, 1)
            if r < alpha:
                RSM_model, best_conf, best_model, k, max_acc, x, y = test_mod(RSM_model, best_conf, best_model, data,
                                                                              hist_acc, hist_res, k, macc, max_acc, s,
                                                                              x, y)
    # plot the results
    print('bestconf',best_conf)
    print('error of RSM',max_acc)
    plt.plot(range(len(macc)), macc)
    plt.plot(range(len(macc)), macc)
    plt.plot(range(len(hist_acc)), hist_acc)
    plt.show()
    return best_conf, best_model, max_acc

# function to test configuration and update paramters
# - train MNIST
# - compare the results
# - add solution to RSM dataset
# - train RSM
# - save configurations
def test_mod(RSM_model, best_conf, best_model, data, hist_acc, hist_res, k, macc, max_acc, s, x, y):
    res = train_MINST.train_model_s(s.get_MNIST(), data)
    if res[0] > max_acc:
        max_acc = res[0]
        best_model = res[1]
        best_conf = s
    (x, y) = add_train((x, y), (s.get_RSM(), res[0]))
    (acc, RSM_model) = learn_RSM(x, y)
    macc.append(max_acc)
    hist_res.append(res[0])
    hist_acc.append(acc)
    np.save('../hyperlearn/hist_macc',macc)
    np.save('../hyperlearn/hist_res',hist_res)
    np.save('../hyperlearn/hist_acc',hist_acc)
    print('RSM_acc', acc)
    k = k + 1
    return RSM_model, best_conf, best_model, k, max_acc, x, y


if __name__ == '__main__':
    res = learn_hyperparam()
    print('bestconf',res[0])
    print('error of RSM',res[3])
    # s=[0, np.array([], dtype='float64'), 0.14289055892459254, 0.0, 0.0, 0.64011881862533493, 0.66598721505367042, 1, 1]
    # print(train_MINST.train_model_s(s, data)[0])
