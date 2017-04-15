from opal.core.io import read_params_from_file, write_measures_to_file
import sys
import json
import subprocess


def run(params, problem):
    """
    Runs the black box with the given parameters.
    """
    with open("hyperMads/config.json") as f:
        js = json.load(f)
    command = "ssh -i {} {}@{} ".format(js["key"], js["user"], js["host"])
    command += "\"python 2>/dev/null ~/hyperNN/problem/runnee.py "

    command += "--noeuds {} {} {} ".format(
        params["n1"], params["n2"], params["n3"])
    del params["n1"], params["n2"], params["n3"]
    params["activation"] = "relu"
    params["n_epoch"] = 100
    params["batch_size"] = 200
    params["nesterov"] = True
    command += " ".join(map(lambda k: "--{} {}".format(k, params[k]), params))

    command += "\""

    out = subprocess.check_output(args=command, shell=True)

    return {'acc': - float(out.strip())}


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
