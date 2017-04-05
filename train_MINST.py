from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model


def train_model_s(s, data):
    activation = s[5]
    n_couches = s[0]
    noeuds = s[1]
    lr = s[2]
    moment = s[3]
    nesterov = s[4]
    return train_model(activation,
                       n_couches,
                       noeuds,
                       lr,
                       moment,
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
        network = Dense(noeuds[i], activation=activation)(network)

    network = Dense(10, activation="softmax")(network)

    model = Model(input=input, output=network)

    opt = keras.optimizers.SGD(lr=learning_rate,
                               momentum=moment,
                               decay=0.0,
                               nesterov=nesterov)

    model.compile(optimizer=opt, loss="categorical_crossentropy")

    model.fit(x_train, y_train,
              nb_epoch=2,
              batch_size=1280,
              verbose=0,
              shuffle=True,
              validation_data=(x_valid, y_valid))

    loss = model.evaluate(x_valid, y_valid, verbose=0)
    print('MINST_loss', loss)
    return (loss, model)
