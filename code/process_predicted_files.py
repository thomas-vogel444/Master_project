import process_results.results_processing_function as rf
from process_results.ExperimentalResults import ExperimentalResults
from dataset_generation.CTScanImage import CTScanImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
import pprint as pp

if __name__ == "__main__":
	# Base parameters
	data_directory 				= "../ct_atrium/testing"
	CT_scan_parameters_template = {
		"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
		"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
		"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
		"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
	}

	test_CT_scans = [CTScanImage("14040204", CT_scan_parameters_template)]

	# experiment_base_directory 	= "../experimental_results/varying_number_of_convolutional_layers"	
	# experiment_names = ["1_conv_layer", "2_conv_layers", "3_conv_layers", "4_conv_layers"]

	# experiment_base_directory 	= "../experimental_results/varying_number_of_connected_layers"	
	# experiment_names = ["1_connected_layer", "2_connected_layers", "3_connected_layers"]

	# experiment_base_directory 	= "../experimental_results/varying_number_of_feature_maps"	
	# experiment_names = ["starting_with_16", "starting_with_32", "starting_with_64"]

	# experiment_base_directory 	= "../experimental_results/varying_number_of_hidden_units"	
	# experiment_names = ["100_hidden_units", "200_hidden_units", "500_hidden_units", "1000_hidden_units"]

	# experiment_base_directory 	= "../experimental_results/varying_learning_rate"	
	# experiment_names = ["0_01", "0_05", "0_1", "0_5"]

	# experiment_base_directory 	= "../experimental_results/varying_momentum"	
	# experiment_names = ["0", "0_05", "0_1", "0_5"]

	# experiment_base_directory 	= "../experimental_results/varying_activation_function"	
	# experiment_names = ["ReLU", "Tanh", "Sigmoid"]

	# experiment_base_directory 	= "../experimental_results/varying_pooling_function"	
	# experiment_names = ["SpatialMaxPooling", "SpatialAveragePooling"]

	# experiment_base_directory 	= "../experimental_results/varying_training_dataset"	
	# experiment_names = ["no_atrium_box", "small_atrium_box", "large_atrium_box"]

	experiment_base_directory 	= "../experimental_results/varying_training_dataset_with_average_pooling"	
	experiment_names = ["no_atrium_box", "small_atrium_box", "large_atrium_box"]

	# experiment_base_directory 	= "../experimental_results/varying_training_size"	
	# experiment_names = ["2500000"]

	#************************************************************************************************************************
	# Get the experimental results	
	different_z = [15, 25, 30]

	all_masks = []
	for experiment_name in experiment_names:
		experiment_path = os.path.join(experiment_base_directory, experiment_name)
		
		experimental_results = ExperimentalResults(experiment_path, test_CT_scans)
		segmented_CT_scans 	 = experimental_results.segmented_CT_scans

		print "=====> Experiment statistics for %s"%experiment_name
		pp.pprint(segmented_CT_scans[0].get_classification_statistics())

		experiment_masks = []
		for z in different_z:
			experiment_masks.append(segmented_CT_scans[0].get_mask(z, "z"))

		all_masks.append(experiment_masks)

	fig = plt.figure()
	for i_experiment, experiment_masks in enumerate(all_masks):
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1)
		plt.imshow(experiment_masks[0])
		plt.axis('off')
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1 + len(all_masks))
		plt.imshow(experiment_masks[1])
		plt.axis('off')
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1 + 2*len(all_masks))
		plt.imshow(experiment_masks[2])
		plt.axis('off')
	plt.show()








