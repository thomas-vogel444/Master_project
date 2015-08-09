import argparse
import os
from experiments.Segmentator import Segmentator

# Parameters
code_path = os.path.abspath("CNN/segment.lua")

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', 	default = '../experimental_results/test_experiment/1')
parser.add_argument('-s', '--dataset_path', default = '../datasets/segmentation_datasets.hdf5')
args = parser.parse_args()

# Segment stuff...
segmentation_parameters = {
		"GPU_id"			: 1,
		"number_of_GPUs"	: 3,
		"segmentationCode"	: code_path,
		"segmentationFile" 	: args.dataset_path,
		"modelDirectory"	: args.model_path
	}

segmentator = Segmentator(segmentation_parameters)
segmentator.segment()
