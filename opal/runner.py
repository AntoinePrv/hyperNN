import sys
sys.path.append(".")

from opal.core.io import read_params_from_file, write_measures_to_file
from problem.load_data import load_data_bis
from problem.train_MINST import train_model


def run(params, problem):
    """
    Runs the black box with the given parameters.
    """
    lr = params['lr']
    return {'acc': lr**2 + 1}


if __name__ == '__main__':
    # Reading paramters provided by OPAL
    param_file = sys.argv[1]   # Name of the file with the bb parameters
    problem = sys.argv[2]      # Problem name
    output_file = sys.argv[3]  # Name of the file to output results to

    # Get the paramters as dictionnary from the file
    params = read_params_from_file(param_file)

    # Run the black box and get the relevant measures
    measures = run(params, problem)

    # Write the measures back to the output file
    write_measures_to_file(output_file, measures)
