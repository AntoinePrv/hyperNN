import sys
sys.path.append(".")

import argparse
from problem.load_data import load_data_bis
from problem.train_MINST import train_model
from logger import custom_logger


def run(**kwargs):
    data = load_data_bis()
    _, acc = train_model(data, **kwargs)


if __name__ == "__main__":
    # create logger
    logger = custom_logger(__name__, "hyperOpal/log/runnee.log")

    run()
