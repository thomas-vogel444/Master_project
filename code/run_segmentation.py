from process_results.Segmentator import Segmentator
import re
import argparse
import os
import functools
from multiprocessing import Pool


def get_CT_scan_number(segmentation_dataset):
        p = re.compile("../datasets/segmentation_dataset_for_(.*)")
        return p.search(segmentation_dataset).group(1)

def segment(segmentation_dataset, segmentation_dataset_directory, predicted_files_directory):
		file_number = int(re.compile("segmentation_dataset_(.*).hdf5").search(segmentation_dataset).group(1))
		available_GPUs = [1,2,3,4]

		# Setting up the segmentation
		segmentation_parameters = {
			"GPU_id"			: available_GPUs[(file_number%len(available_GPUs))],
			"number_of_GPUs"	: 1,
			"segmentationCode"	: os.path.abspath("CNN/segment.lua"),
			"modelDirectory"	: args.model_path,
		}

		segmentator 				= Segmentator(segmentation_parameters)
		segmentation_dataset_path 	= os.path.join(segmentation_dataset_directory, segmentation_dataset)
		predicted_file 				= "predicted_file_%i.hdf5"%file_number
		predicted_path 				= os.path.join(predicted_files_directory, predicted_file)
		
		print "Segmenting %s and storing the predicted results in %s"%(segmentation_dataset, predicted_file)
		height, width = 480, 480
		segmentator.segment(segmentation_dataset_path, predicted_path, width, height)

if __name__ == "__main__":
	# Command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--segmentation_dataset_directory', default = '../datasets/segmentation_dataset_for_14040204')
	parser.add_argument('-m', '--model_path',   default = '../experimental_results/varying_datasets_test/small_atrium_box')

	args = parser.parse_args()

	segmentation_datasets 	= [segmentation_dataset for segmentation_dataset in os.listdir(args.segmentation_dataset_directory)]

	predicted_files_directory = os.path.join(args.model_path, "predicted_files_%s"%get_CT_scan_number(args.segmentation_dataset_directory))
	if not os.path.exists(predicted_files_directory):
		    os.makedirs(predicted_files_directory)

	# Segment all the segmentation files into predicted files
	print "Segmenting files in %s"%predicted_files_directory
	segmentation_function = functools.partial(segment,  segmentation_dataset_directory 	= args.segmentation_dataset_directory, 
														predicted_files_directory 		= predicted_files_directory)

	pool = Pool(processes=4)
	pool.map(segmentation_function, segmentation_datasets)















