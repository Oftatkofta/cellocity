from cellocity import validation
from pathlib import Path

inpath = Path(r"C:\Users\Jens\Documents\_Microscopy\FrankenScope2\Calibration stuff\DIC_truth")
outpath3 = Path(r"C:\Users\Jens\Desktop\temp3")

if __name__ == "__main__":
    validation.run_validation(inpath, outpath3)
