from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model
import sklearn.model_selection as mds
import numpy as np
import math
import load_data
import train_MINST

def learn_RSM(x, y):
    x_train, x_test, y_train, y_test = mds.train_test_split(x, y, test_size=0.10, random_state=0)
    x_train, x_valid, y_train, y_valid = mds.train_test_split(x_train, y_train, test_size=0.10, random_state=0)
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
                     batch_size=50,
                     verbose=0,
                     shuffle=True,
                     validation_data=(x_valid, y_valid))
    return model.evaluate(x_valid,y_valid,verbose=0),model

def transform_s(s):
    # train : [n_couches,c1,c2,c3,c4,c5,lr,moment,nesterov,a1,a2,a3]
    # new : [n_couches/max,noeuds/max,lr,moment,nesterov,a]
    t = np.array([s[0]/5., 0, 0, 0, 0, 0, s[2], s[3], s[4], 0, 0, 0])
    # print(to_train.shape)
    t[9 + s[5]] = 1
    for n in range(len(s[1])):
        t[1 + n] = s[1][n]/500.
    return t

def add_train(old, new):
    to_train = transform_s(new[0])
    if old[0]==None:
        return np.matrix(to_train), [new[1]]
    else:
        # print(old[0].shape,np.matrix(to_train).shape)
        return np.append(old[0], np.matrix(to_train), axis=0), np.append(old[1], [new[1]], axis=0)

def sample_sol():
    r = [0, 0, 0, 0, 0, 0]
    r[0] = np.random.randint(0, 5)
    r[1] = np.random.randint(10, 500, (r[0]))
    r[2] = math.exp(-np.random.uniform(1, 16))
    r[3] = np.random.uniform(0, 1)
    r[4] = np.random.randint(0, 1)
    r[5] = np.random.randint(0, 2)
    # print(r)
    return r

def learn_hyperparam():
    data = load_data()
    x = None
    y = None
    # initialisation
    for i in range(5):
        s = sample_sol()
        res = train_MINST.train_model_s(s,data)
        (x, y) = add_train((x, y), (s, res[0]))
    (loss , RSM_model) = learn_RSM(x, y)

    alpha = 0.01
    min_loss = 1000
    best_model = []
    k = 0

    while k < 200:
        print(k)
        print(x.shape)
        s = sample_sol()
        p = RSM_model.predict(np.matrix(transform_s(s)))
        print('predict_loss ', p, 'min_loss', min_loss)
        if p < min_loss:
            r = np.random.uniform(0, 1)
            if r > alpha:
                res = train_MINST.train_model_s(s, data)
                if res[0] < min_loss:
                    min_loss = res[0]
                    best_model = res[1]
                (x, y) = add_train((x, y), (s, res[0]))
                (loss, RSM_model) = learn_RSM(x, y)
                print('RSM_loss',loss)
                k = k + 1
        else:
            r = np.random.uniform(0, 1)
            if r < alpha:
                res = train_MINST.train_model_s(s,data)
                if res[0] < min_loss:
                    min_loss = res[0]
                    best_model = res[1]
                (x, y) = add_train((x, y), (s, res[0]))
                (loss, RSM_model) = learn_RSM(x, y)
                print('RSM_loss', loss)
                k = k + 1