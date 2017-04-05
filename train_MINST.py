from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model
from keras import regularizers


def train_model_s(s, data):
    activation = s[8]
    n_couches = s[0]
    noeuds = s[1]
    learning_rate = s[2]
    reg_l1= s[3]
    reg_l2= s[4]
    moment = s[5]
    decay = s[6]
    nesterov = s[7]
    return train_model(activation,
                       n_couches,
                       noeuds,
                       learning_rate,
                       reg_l1,
                       reg_l2,
                       moment,
                       decay,
                       nesterov,
                       data)


def train_model(activation,
                n_couches,
                noeuds,
                learning_rate,
                reg_l1,
                reg_l2,
                moment,
                decay,
                nesterov,
                data):
    ((x_train, y_train), (x_valid, y_valid), (x, y)) = data
    dim = x_train.shape[1]
    network = Input(shape=(dim,))
    input = network
    activation = ['sigmoid', 'tanh', 'relu', 'softmax'][activation]

    for i in range(n_couches):
        network = Dense(noeuds[i], activation=activation,W_regularizer=regularizers.l1l2(reg_l1,reg_l2))(network)

    network = Dense(10, activation="softmax",W_regularizer=regularizers.l1l2(reg_l1,reg_l2))(network)

    model = Model(input=input, output=network)

    opt = keras.optimizers.SGD(lr=learning_rate,
                               momentum=moment,
                               decay=decay,
                               nesterov=nesterov)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

    model.fit(x_train, y_train,
              nb_epoch=2,
              batch_size=1280,
              verbose=0,
              shuffle=True,
              validation_data=(x_valid, y_valid))

    loss = model.evaluate(x_valid, y_valid, verbose=0)
    print('MINST_loss', loss)
    return (loss[1], model)
