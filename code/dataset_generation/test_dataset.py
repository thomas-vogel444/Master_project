import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import h5py
import lib.dataset_functions as df

parameters = {
		"data_directory" 		: "../../ct_atrium",
		"CT_scan_path_template" : "../../ct_atrium/CTScan_name",
		"NRRD_path_template"    : "../../ct_atrium/CTScan_name/CTScan_name.nrrd",
		"DICOM_path_template"   : "../../ct_atrium/CTScan_name/DICOMS/DICOM_name",
		"ct_directory_pattern"  : re.compile("[0-9]{8}"),
		"patch_size"    		: 32,
		"n_training_CT_scans"   : 22,
		"n_testing_CT_scans"	: 5,
		"n_training_examples_per_CT_scan" : 5000,
		"n_testing_examples_per_CT_scan"  : 2500
	}

z = 40

# Generate a test dataset from a single DICOM image from a single CT scan
CT_scan 		   = ["14022803"]
CT_scan_directory  = "../../ct_atrium/14022803"
CT_scan_dicoms     = df.get_DICOMs(CT_scan_directory)

CT_scan_dictionary = {CT_scan[0]:CT_scan_dicoms}

print "=======> Generating the testing dataset <======="
generated_dataset, generated_labels = df.generate_random_dataset(CT_scan, CT_scan_dictionary, parameters["n_testing_examples_per_CT_scan"], parameters, z)

# Get the NRRD and CT 3D images
NRRD_path_template 					= "../../ct_atrium/CTScan_name/CTScan_name.nrrd"
nrrd_path 							= NRRD_path_template.replace("CTScan_name", CT_scan[0])
CT_scan_labels, CT_scan_nrrd_header = df.get_NRRD_array(nrrd_path)

DICOM_path_template = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"
CT_scan_image       = df.get_CT_scan_array(CT_scan[0], CT_scan_dicoms, CT_scan_nrrd_header["sizes"], DICOM_path_template)

# Saving the datasets
f 							  = h5py.File("generated_dataset.hdf5", "w")
testing_dataset_hdf5 	  	  = f.create_dataset("testing_dataset", generated_dataset.shape, dtype="uint32")
testing_dataset_hdf5[...]  	  = np.int16(generated_dataset)
testing_labels_hdf5 	  	  = f.create_dataset("testing_labels", generated_labels.shape, dtype="uint8")
testing_labels_hdf5[...]  	  = generated_labels
f.close()

# Plot various patches from the test dataset
dataset_path = "generated_dataset.hdf5"

f = h5py.File(dataset_path, "r")
testing_dataset = f["testing_dataset"]
testing_labels = f["testing_labels"]

i_non_atrium_patch = len(testing_labels)-1
fig = plt.figure()
a   = fig.add_subplot(1,3,1)
plt.imshow(testing_dataset[i_non_atrium_patch,4,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,3,2)
plt.imshow(testing_dataset[i_non_atrium_patch,1,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.show()

i_atrium_patch = 0
fig = plt.figure()
a   = fig.add_subplot(1,3,1)
plt.imshow(testing_dataset[i_atrium_patch,4,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,3,2)
plt.imshow(testing_dataset[i_atrium_patch,1,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.show()

f.close()