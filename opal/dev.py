import sys
sys.path.append(".")

from problem.load_data import load_data_bis
from problem.train_MINST import train_model


if __name__ == "__main__":
    data = load_data_bis()

    print(train_model(data, n_epoch=2))
