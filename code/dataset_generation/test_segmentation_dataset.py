from lib.CTScanImage import CTScanImage
import re
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_directory 				= "../../ct_atrium/"
testing_CT_scans_directory	= os.path.join(data_directory, "testing")
CT_scan_parameters_template = {
				"CT_scan_path_template" : os.path.join(testing_CT_scans_directory, "CTScan_name"),
				"NRRD_path_template"    : os.path.join(testing_CT_scans_directory, "CTScan_name/CTScan_name.nrrd"),
				"DICOM_directory"		: os.path.join(testing_CT_scans_directory, "CTScan_name/DICOMS"),
				"DICOM_path_template"   : os.path.join(testing_CT_scans_directory, "CTScan_name/DICOMS/DICOM_name"),
				"CT_directory_pattern"  : re.compile("[0-9]{8}")
			}
testing_CT_scan_names = [directory for directory in os.listdir(testing_CT_scans_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
testing_CT_scan_name  = testing_CT_scan_names[0]

segmented_CT_scan			  		 	 = CTScanImage(testing_CT_scan_name, CT_scan_parameters_template)
dicom_height, dicom_width, number_dicoms = segmented_CT_scan.image.shape
x_slice, y_slice, z_slice 				 = (250, 250, 30)

# import pdb; pdb.set_trace()

fig = plt.figure()
a   = fig.add_subplot(2,3,1)
plt.imshow(segmented_CT_scan.image[:,:,z_slice], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,4)
plt.imshow(segmented_CT_scan.labels[:,:,z_slice], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,2)
plt.imshow(segmented_CT_scan.image[:,y_slice,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,5)
plt.imshow(segmented_CT_scan.labels[:,y_slice,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,3)
plt.imshow(segmented_CT_scan.image[x_slice,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,6)
plt.imshow(segmented_CT_scan.labels[x_slice,:,:], cmap = cm.Greys_r)
plt.show()

# import the image and labels
segmentation_dataset_path = "../../datasets/segmentation_datasets.hdf5"
f_segmentation = h5py.File(segmentation_dataset_path, "r")
true_labels_fixed_z = f_segmentation["labels_fixed_z"]
true_labels_fixed_y = f_segmentation["labels_fixed_y"]
true_labels_fixed_x = f_segmentation["labels_fixed_x"]

true_values_fixed_z = f_segmentation["values_fixed_z"]
true_values_fixed_y = f_segmentation["values_fixed_y"]
true_values_fixed_x = f_segmentation["values_fixed_x"]

fig = plt.figure()
a   = fig.add_subplot(2,3,1)
plt.imshow(true_values_fixed_z, cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,4)
plt.imshow(true_labels_fixed_z, cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,2)
plt.imshow(true_values_fixed_y, cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,5)
plt.imshow(true_labels_fixed_y, cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,3)
plt.imshow(true_values_fixed_x, cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(2,3,6)
plt.imshow(true_labels_fixed_x, cmap = cm.Greys_r)
plt.show()


f_segmentation.close()