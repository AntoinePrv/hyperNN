from opal.core.io import read_params_from_file, write_measures_to_file
import sys
from logger import custom_logger


def run(params, problem):
    """
    Runs the black box with the given parameters.
    """
    lr = params['lr']
    return {'acc': (lr-.97)**2 + 1}


if __name__ == '__main__':
    # Reading paramters provided by OPAL
    param_file = sys.argv[1]   # Name of the file with the bb parameters
    problem = sys.argv[2]      # Problem name
    output_file = sys.argv[3]  # Name of the file to output results to

    logger = custom_logger(__name__, "hyperOpal/log/runner.log")
    # Get the paramters as dictionnary from the file
    params = read_params_from_file(param_file)
    logger.info(params)

    # Run the black box and get the relevant measures
    measures = run(params, problem)
    logger.info(measures)

    # Write the measures back to the output file
    write_measures_to_file(output_file, measures)
