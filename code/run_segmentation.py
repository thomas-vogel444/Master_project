import re
import argparse
import os
from process_results.Segmentator import Segmentator

def get_slice_number(segmentation_dataset):
	p = re.compile("segmentation_dataset_(.*).hdf5")
	return int(p.search(segmentation_dataset).group(1))

if __name__ == "__main__":
	# Command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--segmentation_dataset_directory', default = '../datasets/segmentation_dataset_for_14040204')
	parser.add_argument('-m', '--model_path',   default = '../experimental_results/varying_datasets_test/small_atrium_box')

	args = parser.parse_args()

	# Setting up the segmentation
	segmentation_parameters = {
			"GPU_id"			: 1,
			"number_of_GPUs"	: 1,
			"segmentationCode"	: os.path.abspath("CNN/segment.lua"),
			"modelDirectory"	: args.model_path,
		}

	segmentator 			= Segmentator(segmentation_parameters)
	segmentation_datasets 	= [segmentation_dataset for segmentation_dataset in os.listdir(args.segmentation_dataset_directory)]
	height, width 			= 480, 480

	predicted_files_directory = os.path.join(args.model_path, "predicted_files")
	if not os.path.exists(predicted_files_directory):
		    os.makedirs(predicted_files_directory)

	# Segment all the segmentation files into predicted files
	for segmentation_dataset in segmentation_datasets:
		segmentation_dataset_path 	= os.path.join(args.segmentation_dataset_directory, segmentation_dataset)
		predicted_file 				= "predicted_file_%i.hdf5"%get_slice_number(segmentation_dataset)
		predicted_path 				= os.path.join(predicted_files_directory, predicted_file)
		
		print "Segmenting %s and storing the predicted results in %s"%(segmentation_dataset, predicted_file)
		segmentator.segment(segmentation_dataset_path, predicted_path, width, height)