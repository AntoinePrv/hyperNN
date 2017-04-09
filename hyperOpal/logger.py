import logging
import os
import errno


def create_file(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
            open(filename, 'a').close()
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def custom_logger(name, file):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    create_file(file)
    ch = logging.FileHandler(file)
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger
