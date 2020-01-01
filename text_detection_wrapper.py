## Creating text_detection_wrapper.py

import os
import argparse
from text_detection import detect_fun

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", type=str,
	help="path to input directory")
ap.add_argument("-f", "--out", type=str,
   help='path to output directory')
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument( '-cr', "--crop", action='store_true')
args = vars(ap.parse_args())

for img in sorted(os.listdir(args['directory'])):
  detect_fun(os.path.join(args['directory'],img), args['out'], args['crop'], args['min_confidence'])
