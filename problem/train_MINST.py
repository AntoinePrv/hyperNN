from keras.layers import Input, Dense
import keras.optimizers
from keras.models import Model
from keras import regularizers
import logging


def train_model_s(s, data):
    return train_model(
        data,
        activation=['sigmoid', 'tanh', 'relu', 'softmax'][s[8]],
        n_couches=s[0],
        noeuds=s[1],
        learning_rate=s[2],
        reg_l1=s[3],
        reg_l2=s[4],
        moment=s[5],
        decay=s[6],
        nesterov=s[7],
    )


def train_model(data, **kwargs):
    hparams = {
        "activation":    "relu",
        "n_couches":     1,
        "noeuds":        [50],
        "learning_rate": .01,
        "reg_l1":        .001,
        "reg_l2":        .001,
        "moment":        .01,
        "decay":         1e-6,
        "nesterov":      True,
        "n_epoch":       100,
        "batch_size":    200
    }
    hparams.update(kwargs)

    x_train = data["X_train"]
    y_train = data["Y_train"]
    x_valid = data["X_valid"]
    y_valid = data["Y_valid"]
    dim = x_train.shape[1]
    network = Input(shape=(dim,))
    input = network

    for i in range(hparams["n_couches"]):
        network = Dense(
            hparams["noeuds"][i],
            activation=hparams["activation"],
            W_regularizer=regularizers.l1_l2(
                hparams["reg_l1"],
                hparams["reg_l2"]
            )
        )(network)

    network = Dense(
        10, activation="softmax",
        W_regularizer=regularizers.l1_l2(
            hparams["reg_l1"],
            hparams["reg_l2"])
    )(network)

    model = Model(input=input, output=network)

    opt = keras.optimizers.SGD(lr=hparams["learning_rate"],
                               momentum=hparams["moment"],
                               decay=hparams["decay"],
                               nesterov=hparams["nesterov"])

    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              nb_epoch=hparams["n_epoch"],
              batch_size=hparams["batch_size"],
              verbose=0,
              shuffle=True,
              validation_data=(x_valid, y_valid))

    loss = model.evaluate(x_valid, y_valid, verbose=0)

    logger = logging.getLogger(__name__)
    logger.info("Acc={}, Params={}".format(loss[1], hparams))

    return (loss[1], model)
