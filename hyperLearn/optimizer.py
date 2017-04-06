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


def learn_RSM(x, y):
    # x_train, x_test, y_train, y_test = mds.train_test_split(x, y, test_size=0.10, random_state=0)
    x_train, x_valid, y_train, y_valid = mds.train_test_split(x, y, test_size=0.10, random_state=0)
    # print('RSMdata',x_train,y_train)

    dim = x.shape[1]
    input = Input(shape=(dim,))
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


def add_train(old, new):
    to_train = new[0]
    if old[0] == None:
        return np.matrix(to_train), [new[1]]
    else:
        # print(old[0].shape,np.matrix(to_train).shape)
        return np.append(old[0], np.matrix(to_train), axis=0), np.append(old[1], [new[1]], axis=0)


def learn_hyperparam():
    data = load_data.load_data()
    x = None
    y = None
    # initialisation
    for i in range(5):
        s = sp.sample()
        print(s.get_MNIST())
        res = train_MINST.train_model_s(s.get_MNIST(), data)
        (x, y) = add_train((x, y), (s.get_RSM(), res[0]))
    (acc, RSM_model) = learn_RSM(x, y)

    hist_acc = []
    alpha = 0.0001
    max_acc = max(y)
    best_model = []
    best_conf = []
    k = 0
    s = sp.sample()
    while k < 200:
        print('n° test', k)
        # print('shape x',x.shape)
        if np.random.uniform(0, 1) < alpha * 0.001:
            s = best_conf
        s = s.gaussian_samp()
        p = RSM_model.predict(np.matrix(s.get_RSM()))
        print('predict_loss ', p, 'acc', max_acc)
        if p > max_acc:
            r = np.random.uniform(0, 1)
            if r > alpha:
                res = train_MINST.train_model_s(s.get_MNIST(), data)
                if res[0] > max_acc:
                    max_acc = res[0]
                    best_model = res[1]
                    best_conf = s
                (x, y) = add_train((x, y), (s.get_RSM(), res[0]))
                (acc, RSM_model) = learn_RSM(x, y)
                hist_acc.append(acc)
                print('RSM_acc', acc)
                k = k + 1
        else:
            r = np.random.uniform(0, 1)
            if r < alpha:
                res = train_MINST.train_model_s(s.get_MNIST(), data)
                if res[0] < max_acc:
                    acc = res[0]
                    best_model = res[1]
                    best_conf = s
                (x, y) = add_train((x, y), (s.get_RSM(), res[0]))
                (acc, RSM_model) = learn_RSM(x, y)
                hist_acc.append(acc)
                print('RSM_acc', acc)
                k = k + 1
    plt.plot(range(acc.shape[0]), acc)
    plt.show()
    return best_conf, best_model, acc


if __name__ == '__main__':
    learn_hyperparam()
    # data = load_data.load_data()
    # s=[0, np.array([], dtype='float64'), 0.14289055892459254, 0.0, 0.0, 0.64011881862533493, 0.66598721505367042, 1, 1]
    # print(train_MINST.train_model_s(s, data)[0])
