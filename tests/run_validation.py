import sys, getopt
from cellocity import validation
from pathlib import Path


if __name__ == "__main__":
    validation.run_validation(inpath, outpath3)
def main(argv):

    inputfolder = ''
    outputfolder = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["infolder=","outfolder="])
    except getopt.GetoptError:
        print ('validation.py -i <infolder> -o <outfolder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('validation.py -i <infolder> -o <outfolder>')
            sys.exit()
        elif opt in ("-i", "--infolder"):
            inpath = Path(arg)
        elif opt in ("-o", "--outfolder"):
            outpath = Path(arg)

    run_validation(inpath, outpath)

if __name__ == "__main__":
   main(sys.argv[1:])
