import sys
sys.path.append("..")
from CTScanImage import CTScanImage
import dataset_functions as df
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import numpy as np


training_data_directory = "../../../ct_atrium/training/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : training_data_directory + "CTScan_name",
		"NRRD_path_template"    : training_data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: training_data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : training_data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_name 	= "14032003" 
xy_padding  	= 50
z_padding   	= 10
CT_scan 		= CTScanImage(CT_scan_name, CT_scan_parameters_template, xy_padding, z_padding)
x, y, z 		= 250, 250, 30

fig = plt.figure()
a   = fig.add_subplot(1,3,1)
plt.imshow(CT_scan.labels_with_atrium_box[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(1,3,2)
plt.imshow(CT_scan.labels_with_atrium_box[:,y,:], cmap = cm.Greys_r)
a   = fig.add_subplot(1,3,3)
plt.imshow(CT_scan.labels_with_atrium_box[x,:,:], cmap = cm.Greys_r)
plt.show()

# Generate a dataset 
n_examples_per_CT_scan_per_label 	= (2, 2, 2)
sampling_type 						= "With_Atrium_Box"
patch_size 							= 32

# Generate a test dataset from a single DICOM image from a single CT scan
print "=======> Generating the testing dataset <======="
# generate_random_dataset(CT_scan_names, n_examples_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=None, xy_padding=0, z_padding=0)
generated_dataset, generated_labels = df.generate_random_dataset([CT_scan_name], n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=z)
print "Expected number of examples generated: %s" %(sum(n_examples_per_CT_scan_per_label))
print "Actual number of examples generated: %s" %(len(generated_labels))
print "Number of examples in the Atrium and non-Atrium: %i, %i" %(len(np.where(generated_labels == 2)[0]), len(np.where(generated_labels == 1)[0]))

i_non_bd_non_atrium_patch = 0
i_bd_non_atrium_patch 	  = 2
i_atrium_patch 			  = 4

fig = plt.figure()
a   = fig.add_subplot(2,4,1)
plt.imshow(generated_dataset[i_non_bd_non_atrium_patch,3,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,5)
plt.imshow(generated_dataset[i_non_bd_non_atrium_patch,0,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,2)
plt.imshow(generated_dataset[i_bd_non_atrium_patch,3,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,6)
plt.imshow(generated_dataset[i_bd_non_atrium_patch,0,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,3)
plt.imshow(generated_dataset[i_atrium_patch,3,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,7)
plt.imshow(generated_dataset[i_atrium_patch,0,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,4)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,8)
plt.imshow(CT_scan.labels_with_atrium_box[:,:,z], cmap = cm.Greys_r)
plt.show()












