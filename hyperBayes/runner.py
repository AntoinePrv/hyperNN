import json
import subprocess


def run(params):
    with open("../hyperOpal/config.json") as f:
        js = json.load(f)
    command = "ssh -i {} {}@{} ".format(js["key"], js["user"], js["host"])
    command += "\"python 2>/dev/null ~/hyperNN/problem/runnee.py "

    command += "--noeuds {} {} ".format(
        params["noeuds"][0], params["noeuds"][1])
    del params["noeuds"]
    command += " ".join(map(lambda k: "--{} {}".format(k, params[k]), params))
    command += "\""

    out = subprocess.check_output(args=command, shell=True)

    return - float(out.strip())


def main(job_id, params):
    for p in ["learning_rate", "reg_l1", "reg_l2", "moment", "decay"]:
        params[p] = float(params[p])
    params["noeuds"] = [int(params["noeuds1"]), int(params["noeuds2"])]
    del params["noeuds1"]
    del params["noeuds2"]

    params.update({
        "activation": "relu",
        "nesterov":   True,
        "batch_size": 200,
        "n_epoch":    100})
    return run(params)
