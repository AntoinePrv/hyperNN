import json
from subprocess import Popen


if __name__ == "__main__":
    with open("hyperRand/config.json") as f:
        js = json.load(f)

    for val in xrange(js["host-range"][0], js["host-range"][1] + 1):
        ip = js["host-template"].replace("*", str(val))
        command = "ssh -i {} {}@{} ".format(js["key"], js["user"], ip)
        command += "\"python ~/hyperNN/hyperRand/runnee.py 20\""

        try:
            Popen(args=command, shell=True)
        except Exception as e:
            print("Error on {}: {}".format(ip, e))
