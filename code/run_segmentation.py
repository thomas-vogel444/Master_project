import argparse
import os
from experiments.Segmentator import Segmentator

if __name__ == "__main__":
	# Command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--segmentation_dataset_directory', default = '../datasets/segmentation_dataset_for_14040204')
	parser.add_argument('-m', '--model_path',   default = '../experimental_results/varying_momentum/0_5')

	args = parser.parse_args()

	# Setting up the segmentation
	segmentation_parameters = {
			"GPU_id"			: 1,
			"number_of_GPUs"	: 4,
			"segmentationCode"	: os.path.abspath("CNN/segment.lua"),
			"modelDirectory"	: args.model_path,
		}

	segmentator 			= Segmentator(segmentation_parameters)
	segmentation_datasets 	= [segmentation_dataset for segmentation_dataset in os.listdir(segmentation_dataset_directory)]
	height, width 			= 480, 480
	print segmentation_datasets

	for index, segmentation_dataset in enumerate(segmentation_datasets):
		segmentation_dataset_path 	= os.path.join(segmentation_dataset, segmentation_dataset_directory)
		predicted_file 				= "predicted_file_%i.hdf5"%index

		segmentator.segment(segmentation_dataset_path, predicted_file, width, height)