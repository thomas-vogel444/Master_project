import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import h5py
from lib.CTScanImage import CTScanImage
import lib.dataset_functions as df

data_directory = "../../ct_atrium/Training/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

n_examples_per_CT_scan_per_label 	= (2, 2)
sampling_type						= "Without_Atrium_Box"
patch_size 							= 32
z 									= 30

# Generate a test dataset from a single DICOM image from a single CT scan
CT_scan_names = ["14032003", "13061101"]

print "=======> Generating the testing dataset <======="
generated_dataset, generated_labels = df.generate_random_dataset(CT_scan_names, n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, sampling_type, z)

print "Expected number of examples generated: %s" %(sum(n_examples_per_CT_scan_per_label)*len(CT_scan_names))
print "Actual number of examples generated: %s" %(len(generated_labels))
print "Generated labels: "
print generated_labels

# Saving the datasets
f 							  = h5py.File("generated_dataset.hdf5", "w")
testing_dataset_hdf5 	  	  = f.create_dataset("testing_dataset", generated_dataset.shape, dtype="uint32")
testing_dataset_hdf5[...]  	  = np.int16(generated_dataset)
testing_labels_hdf5 	  	  = f.create_dataset("testing_labels", generated_labels.shape, dtype="uint8")
testing_labels_hdf5[...]  	  = generated_labels
f.close()

# Plot various patches from the test dataset
CT_scan_0 	 = CTScanImage(CT_scan_names[0], CT_scan_parameters_template)
CT_scan_1 	 = CTScanImage(CT_scan_names[1], CT_scan_parameters_template)
dataset_path = "generated_dataset.hdf5"

f 				= h5py.File(dataset_path, "r")
testing_dataset = f["testing_dataset"]
testing_labels  = f["testing_labels"]

i_atrium_patch 	   = 0
i_non_atrium_patch = len(testing_labels)-1

fig = plt.figure()
a   = fig.add_subplot(2,4,1)
plt.imshow(testing_dataset[i_atrium_patch,3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,5)
plt.imshow(testing_dataset[i_atrium_patch,0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,2)
plt.imshow(testing_dataset[i_non_atrium_patch,3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,6)
plt.imshow(testing_dataset[i_non_atrium_patch,0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,3)
plt.imshow(CT_scan_0.image[:,:,z], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,7)
plt.imshow(CT_scan_0.labels[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,4,4)
plt.imshow(CT_scan_1.image[:,:,z], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,4,8)
plt.imshow(CT_scan_1.labels[:,:,z], cmap = cm.Greys_r)
plt.show()

f.close()