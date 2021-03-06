import sys
sys.path.append("..")
from CTScanImage import CTScanImage
import dataset_functions as df
import re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os


n_examples_per_CT_scan_per_label 	= (2, 2, 2)
sampling_type						= "With_Atrium_Box"
patch_size 							= 32
z 									= 30

# Generate a test dataset from a single DICOM image from a single CT scan
data_directory = "../../../ct_atrium/testing/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
CT_scan_name  = CT_scan_names[0]

CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template)

print "=======> Generating the testing dataset <======="
generated_dataset, generated_labels = df.generate_random_dataset([CT_scan_name], n_examples_per_CT_scan_per_label, CT_scan_parameters_template, patch_size, sampling_type, dicom_index=z)

print "Expected number of examples generated: %s" %(sum(n_examples_per_CT_scan_per_label))
print "Actual number of examples generated: %s" %(len(generated_labels))
print "Generated labels: "
print generated_labels

# Saving the datasets
dataset_path = "test_dataset.hdf5"

f 							  = h5py.File(dataset_path, "w")
testing_dataset_hdf5 	  	  = f.create_dataset("testing_dataset", generated_dataset.shape, dtype="float32")
testing_dataset_hdf5[...]  	  = generated_dataset
testing_labels_hdf5 	  	  = f.create_dataset("testing_labels", generated_labels.shape, dtype="uint8")
testing_labels_hdf5[...]  	  = generated_labels
f.close()

# Plot various patches from the test dataset
f 				= h5py.File(dataset_path, "r")
testing_dataset = np.array(f["testing_dataset"])
testing_labels  = np.array(f["testing_labels"])
f.close()
os.remove(dataset_path)

i_atrium_patch 	   = 0
i_non_atrium_patch = len(testing_labels)-1

fig = plt.figure()
a   = fig.add_subplot(2,3,1)
plt.imshow(testing_dataset[i_atrium_patch,3,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,4)
plt.imshow(testing_dataset[i_atrium_patch,0,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,2)
plt.imshow(testing_dataset[i_non_atrium_patch,3,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,5)
plt.imshow(testing_dataset[i_non_atrium_patch,0,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,3)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,6)
plt.imshow(CT_scan.labels[:,:,z], cmap = cm.Greys_r)
plt.show()
