import process_results.results_processing_function as rf
from process_results.ExperimentalResults import ExperimentalResults
from dataset_generation.CTScanImage import CTScanImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import numpy as np
import os
import re

experiment_path 	= "../experimental_results/varying_training_dataset/small_atrium_box"	
CT_scan_names = ["14022801", "14012303", "14031001", "14031201", "14040204", "14051403", "14051404"]
# CT_scan_names = ["14022801"]

data_directory 				= "../ct_atrium/testing"
CT_scan_parameters_template = {
	"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
	"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
	"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
	"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
	"CT_directory_pattern"  : re.compile("[0-9]{8}")
}

segmented_CT_scans   = [CTScanImage(CT_scan_name, CT_scan_parameters_template) for CT_scan_name in CT_scan_names]
experimental_results = ExperimentalResults(experiment_path, segmented_CT_scans)
segmented_CT_scans 	 = experimental_results.segmented_CT_scans

def get_test_statistics(segmented_CT_scans, test_statistic):
	return [segmented_CT_scan.classification_statistics[test_statistic] \
							for segmented_CT_scan in segmented_CT_scans]

sensitivities 		= get_test_statistics(segmented_CT_scans, "Sensitivity")
specificities 		= get_test_statistics(segmented_CT_scans, "Specificity")
dice_coefficients 	= get_test_statistics(segmented_CT_scans, "Dice Coefficient")

def print_statistics(statistics, test_statistic):
	print "=======> For %s <======="%test_statistic
	print "Mean: %.3f"%round(np.mean(statistics),3)
	print "Standard Deviation: %.5f"%round(np.std(statistics),5)
	print "Minimum: %.3f"%round(min(statistics),3)
	print "Maximum: %.3f"%round(max(statistics),3)

print_statistics(sensitivities, "Sensitivity")
print_statistics(specificities, "Specificity")
print_statistics(dice_coefficients, "Dice Coefficient")

def plot_masks(mask_z, mask_y, mask_x, save=False, filename=None):
	fig = plt.figure()
	gs  = gridspec.GridSpec(1, 3, width_ratios=[3, 1, 1])
	a   = fig.add_subplot(gs[0])
	plt.imshow(mask_z, cmap=cm.Greys_r)
	plt.axis('off')
	a   = fig.add_subplot(gs[1])
	plt.imshow(mask_y.resize((c_height/3, c_height)), cmap=cm.Greys_r)
	plt.axis('off')
	a   = fig.add_subplot(gs[2])
	plt.imshow(mask_x.resize((s_height/3, s_height)), cmap=cm.Greys_r)
	plt.axis('off')
	if save == True:
		plt.savefig(filename, bbox_inches='tight')
	else:
		plt.show()


scale 	= 3
# indices = [[200, 200, 20]]
indices = [[220, 220, 25], [250, 250, 30]]
for i in range(len(segmented_CT_scans)):
	print "For CT scan %s"%segmented_CT_scans[i].CT_scan.name
	for i_index, index in enumerate(indices):
		x, y, z = index
		t_height, t_width = segmented_CT_scans[i].CT_scan.image[:,:,z].shape
		c_height, c_width = segmented_CT_scans[i].CT_scan.image[:,y,:].shape
		s_height, s_width = segmented_CT_scans[i].CT_scan.image[x,:,:].shape

		mask_z = segmented_CT_scans[i].get_mask(z, "z") 
		mask_y = segmented_CT_scans[i].get_mask(y, "y")
		mask_x = segmented_CT_scans[i].get_mask(x, "x")
		
		plot_masks(mask_z, mask_y, mask_x, save = True, filename = 'Masks_for_%s_%i.png'%(segmented_CT_scans[i].CT_scan.name, i_index))














