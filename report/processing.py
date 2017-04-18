import json
import numpy as np


def accuracy(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    accuracy = np.array(list(map(lambda j: j["Accuracy"], data)))
    return accuracy


def params(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data
