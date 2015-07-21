import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lib.CTScanImage as CTScanImage
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lib.dataset_functions as df

data_directory = "../../ct_atrium/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_name = "14032003" 
CT_scan = CTScanImage.CTScanImage(CT_scan_name, CT_scan_parameters_template)

x, y, z 	= 250, 250, 30
xy_padding  = 20
z_padding   = 10
labels_with_atrium_box = CT_scan.get_labels_with_atrium_box(xy_padding, z_padding)

fig = plt.figure()
a   = fig.add_subplot(1,3,1)
plt.imshow(labels_with_atrium_box[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(1,3,2)
plt.imshow(labels_with_atrium_box[:,y,:], cmap = cm.Greys_r)
a   = fig.add_subplot(1,3,3)
plt.imshow(labels_with_atrium_box[x,:,:], cmap = cm.Greys_r)
plt.show()

# Generate a dataset 
n_examples_per_CT_scan_per_label = (2, 2, 2)
patch_size = 32

# Generate a test dataset from a single DICOM image from a single CT scan
CT_scan_names = ["14032003"]

print "=======> Generating the testing dataset <======="
generated_dataset, generated_labels = df.generate_random_dataset(CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, z, xy_padding, z_padding)
print "Expected number of examples generated: %s" %(sum(n_examples_per_CT_scan_per_label))
print "Actual number of examples generated: %s" %(len(generated_labels))
print "Number of examples in the Atrium and non-Atrium: %i, %i" %(len(np.where(generated_labels == 2)[0]), len(np.where(generated_labels == 1)[0]))

i_non_bd_non_atrium_patch = 0
i_bd_non_atrium_patch 	  = 2
i_atrium_patch 			  = 4

fig = plt.figure()
a   = fig.add_subplot(2,4,1)
plt.imshow(generated_dataset[i_non_bd_non_atrium_patch,3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,5)
plt.imshow(generated_dataset[i_non_bd_non_atrium_patch,0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,2)
plt.imshow(generated_dataset[i_bd_non_atrium_patch,3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,6)
plt.imshow(generated_dataset[i_bd_non_atrium_patch,0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,3)
plt.imshow(generated_dataset[i_atrium_patch,3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,7)
plt.imshow(generated_dataset[i_atrium_patch,0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,4)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,8)
plt.imshow(labels_with_atrium_box[:,:,z], cmap = cm.Greys_r)
plt.show()

123995 + 11005
5391 + 129609












