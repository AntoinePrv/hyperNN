from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model
import sklearn.model_selection as mds
import numpy as np
import math
import load_data
import train_MINST

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
    return model.evaluate(x_valid,y_valid,verbose=0),model

def t_s(s):
    # new : [n_couches, noeuds, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, activation]
    # train : [n_couches, c1, c2, c3, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, a1, a2, a3]
    t = np.array([s[0]/3., 0, 0, 0, s[2], s[3], s[4], s[5], s[6], s[7], 0, 0, 0])
    t[10 + s[8]] = 1
    for n in range(len(s[1])):
        t[1 + n] = s[1][n]/500.
    return t

def t_inv_s(t):
    s = np.array([t[0]*3., [], t[4], t[5], t[6], t[7], t[8], t[9], 0])
    noeuds = []
    if t[1]!=0 : noeuds.append(t[1])*500
    if t[2]!=0 : noeuds.append(t[2])*500
    if t[3]!=0 : noeuds.append(t[3])*500
    s[1]=np.array(noeuds)
    for i in range(3):
        if t[i+10]==1:s[8]=i
    return s

def add_train(old, new):
    to_train = t_s(new[0])
    if old[0]==None:
        return np.matrix(to_train), [new[1]]
    else:
        # print(old[0].shape,np.matrix(to_train).shape)
        return np.append(old[0], np.matrix(to_train), axis=0), np.append(old[1], [new[1]], axis=0)

def sample_sol_uniform():
    r = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    r[0] = np.random.randint(4)              # n_couches
    r[1] = np.random.randint(1, 500, (r[0]))    # noeuds
    r[2] = math.exp(-np.random.uniform(1, 16))  # learning_rate,
    r[3] = np.random.uniform(0, 1)              # reg_l1
    r[4] = np.random.uniform(0, 1)              # reg_l2
    r[5] = np.random.uniform(0, 1)              # moment
    r[6] = np.random.uniform(0, 1)              # decay
    r[7] = np.random.randint(2)              # nesterov
    r[8] = np.random.randint(3)              # activation
    # print(r)
    return r

def sample_sol(pt):
    # [n_couches, noeuds, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, activation]
    # [n_couches, c1, c2, c3, learning_rate, reg_l1, reg_l2, moment, decay, nesterov, a1, a2, a3]:13
    # pt = x[np.random.randint(x.shape[0])]
    vois = 0.1
    new = np.random.normal(loc = pt,scale = np.array([0.5,vois,vois,vois,pt[4]*vois,pt[5]*vois,pt[6]*vois,pt[7]*vois,pt[8]*vois,0.5,0.5,0.5,0.5]))
    # print(new[0]*3+0.5)
    for i in range(new.shape[0]):
        new[i]=max(0,new[i])
    res = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    res[0] = min(3, int(new[0]*3+0.5))    # n_couches
    noeuds = []
    for i in range(res[0]):
        noeuds.append(max(10,min(500, int(new[1+i]*500))))
    res[1] = np.array(noeuds)       # noeuds
    res[2] = min(new[4],1)          # learning_rate
    res[3] = min(new[5],1)          # reg_l1
    res[4] = min(new[6],1)          # reg_l2
    res[5] = min(new[7],1)          # moment
    res[6] = min(new[8],1)          # decay
    res[7] = min(int(2*new[9]),1)   # nesterov
    m = new[10]
    res[8] = 0
    for i in range(2):
        if new[i+11]>m:
            m=new[i+11]
            res[8]=i+1
    print(res)
    return res

def learn_hyperparam():
    data = load_data.load_data()
    x = None
    y = None
    # initialisation
    for i in range(5):
        s = sample_sol_uniform()
        res = train_MINST.train_model_s(s,data)
        (x, y) = add_train((x, y), (s, res[0]))
    (acc , RSM_model) = learn_RSM(x, y)

    alpha = 0.001
    max_acc = max(y)
    best_model = []
    best_conf = []
    k = 0

    while k < 50:
        print('n° test',k)
        print('shape x',x.shape)
        if np.random.uniform(0, 1) <0.001:
            s=best_conf
        s = sample_sol(t_s(s))
        p = RSM_model.predict(np.matrix(t_s(s)))
        print('predict_loss ', p, 'acc', max_acc)
        if p > max_acc:
            r = np.random.uniform(0, 1)
            if r > alpha:
                res = train_MINST.train_model_s(s, data)
                if res[0] > max_acc:
                    max_acc = res[0]
                    best_model = res[1]
                    best_conf=s
                (x, y) = add_train((x, y), (s, res[0]))
                (acc, RSM_model) = learn_RSM(x, y)
                print('RSM_acc',acc)
                k = k + 1
        else:
            r = np.random.uniform(0, 1)
            if r < alpha:
                res = train_MINST.train_model_s(s,data)
                if res[0] < max_acc:
                    acc = res[0]
                    best_model = res[1]
                    best_conf = s
                (x, y) = add_train((x, y), (s, res[0]))
                (loss, RSM_model) = learn_RSM(x, y)
                print('RSM_acc', loss)
                k = k + 1
    return best_conf,best_model,acc

if __name__=='__main__':
    learn_hyperparam()
    # p = sample_sol_uniform()
    # print(p)
    # t = t_s(p)
    # print(sample_sol(t))
    # print(sample_sol(t))
    # print(sample_sol(t))
    # print(sample_sol(t))
    # print(sample_sol(t))
    # print(sample_sol(t))