


import argparse
import importlib
from matplotlib import pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mlp
from cycler import cycler
import glob

#Default figure parameters
params = {'legend.fontsize': 'large',
          'figure.figsize': (8, 4),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'legend.loc': 'lower right',
         'axes.prop_cycle': mlp.cycler(color=['#0000a7', '#c1272d', '#028A0F'])}

pylab.rcParams.update(params)

if __name__ == "__main__":

    """
    Simple script to generate plots from the command line:

    python run.py -n "figure_name" 

    will generate and save the chosen plot using the proper plot.py script

    Inputting  "all" as figure_name will generate and save all plots without showing them

    All figures are saved in the "all_figures" folder, as well as their respectable script folders.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-n", type=str, required=True)
    parser.add_argument("-s", type=str, default="true")
    args = parser.parse_args()

    if args.n != "all":
        fig_module = importlib.import_module(args.n + ".plot")
        s = args.s.lower()
        if s == "false":
            fig_module.run(show = False)
        else:
            fig_module.run(show = True)

    else:
        print("Generating all plots:")
        for n in sorted(glob.glob("*")):
            if n not in ["__pycache__", "all_figures", "run.py", "reformat_data.py"]:
                print(n)
                fig_module = importlib.import_module(n + ".plot")
                fig_module.run(show = False)
