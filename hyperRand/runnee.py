import sys
import os
dir = "/".join(sys.argv[0].split("/")[:-2])
if dir != "":
    os.chdir(dir)
sys.path.append(".")

import numpy as np
from problem.load_data import load_data_bis
from problem.train_MINST import train_model
from problem.logger import custom_logger


def run(**kwargs):
    data = load_data_bis()
    acc, _ = train_model(data, **kwargs)
    return acc


if __name__ == "__main__":
    # create logger
    logger = custom_logger("problem.train_MINST", "hyperRand/log/runnee.log")

    data = load_data_bis()
    n = 1 if len(sys.argv) <= 1 else int(sys.argv[1])

    activation = np.array(["relu", "softmax", "tanh", "sigmoid"])
    activation = activation[np.random.randint(0, 4, n)]
    n_couches = 2  # using only two layer to compare with hyperOpal
    noeuds = np.random.poisson(200, n_couches*n).reshape((n, n_couches))
    learning_rate = np.exp(np.random.normal(-2, 3, n))
    reg_l1 = np.exp(np.random.normal(-2, 3, n))
    reg_l2 = np.exp(np.random.normal(-2, 3, n))
    decay = np.exp(np.random.normal(-2, 3, n))
    moment = np.exp(np.random.normal(-2, 3, n))
    nesterov = np.array([True, False])
    nesterov = nesterov[np.random.randint(0, 2, n)]

    for i in xrange(n):
        run(**{
            "activation":    activation[i],
            "n_couches":     n_couches,
            "noeuds":        noeuds[i].tolist(),
            "learning_rate": learning_rate[i],
            "reg_l1":        reg_l1[i],
            "reg_l2":        reg_l2[i],
            "moment":        moment[i],
            "decay":         decay[i],
            "nesterov":      bool(nesterov[i])})
