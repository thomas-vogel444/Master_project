import process_results.results_processing_function as rf
from process_results.ExperimentalResults import ExperimentalResults
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import re
import pprint as pp

if __name__ == "__main__":
	# Base parameters
	experiment_base_directory 	= "../experimental_results/varying_number_of_convolutional_layers"	
	experiment_names = ["1_conv_layer", "2_conv_layers", "3_conv_layers"]

	# Get the experimental results	
	z = 30

	all_masks = []
	for experiment_name in experiment_names:
		experiment_path = os.path.join(experiment_base_directory, experiment_name)
		
		experimental_results = ExperimentalResults(experiment_path)
		
		print "=====> Experiment statistics for %s"%experiment_name
		pp.pprint(experimental_results.get_classification_statistics())

		experiment_masks = []
		for predicted_labels, segmented_CT_scan in zip(experimental_results.predicted_labels, experimental_results.segmented_CT_scans):
			experiment_masks.append(rf.get_mask(predicted_labels[:,:,z], 
												segmented_CT_scan.labels[:,:,z],
												segmented_CT_scan.image[:,:,z]))
		all_masks.append(experiment_masks)

	# Print the masks
	fig = plt.figure()
	for i_experiment, experiment_masks in enumerate(all_masks):
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1)
		a.set_title(experiment_names[i_experiment].replace("_", " "))
		plt.imshow(experiment_masks[0])
		plt.axis('off')
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1 + len(all_masks))
		plt.imshow(experiment_masks[1])
		plt.axis('off')
		a = fig.add_subplot(len(experiment_masks), len(all_masks), i_experiment + 1 + 2*len(all_masks))
		plt.imshow(experiment_masks[2])
		plt.axis('off')
	plt.show()






