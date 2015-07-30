import os
from Segmentator import Segmentator

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--code_path', 	default = '../CNN/segment.lua')
parser.add_argument('-m', '--model_path', 	default = '../../experimental_results/test_experiment/0')
parser.add_argument('-s', '--dataset_path', default = '../../datasets/segmentation_datasets.hdf5')
args = parser.parse_args()

# Segment stuff...
segmentator = Segmentator(args.dataset_path, args.code_path)
segmentator.segment(args.model_path)