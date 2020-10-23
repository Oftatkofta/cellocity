import sys, getopt
from cellocity import validation
from pathlib import Path

"""
This script runs the validation suite for FiveSigmaAnalysis. It takes a long time to run fully.
"""


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["infolder=", "outfolder="])
    except getopt.GetoptError:
        print('python run_5sigma_validation.py -i <infolder> -o <outfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('python run_5sigma_validation -i <infolder> -o <outfolder>')
            sys.exit()
        elif opt in ("-i", "--infolder"):
            inpath = Path(arg)
        elif opt in ("-o", "--outfolder"):
            outpath = Path(arg)

    validation.run_5sigma_validation(inpath, outpath)


if __name__ == "__main__":
    main(sys.argv[1:])
