


import argparse
import importlib
from matplotlib import pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-s", type=bool, default=False)
    args = parser.parse_args()

    fig_module = importlib.import_module(args.n + ".plot")
    fig_module.run()


