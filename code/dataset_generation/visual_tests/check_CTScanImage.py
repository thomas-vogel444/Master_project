import sys
sys.path.append("..")
from CTScanImage import CTScanImage
import utils
import re
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


data_directory = "../../../ct_atrium/testing/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
CT_scan_name  = CT_scan_names[1]

# CT_scan_name = "14012303"
CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template)

# Plot a bunch of cross sections from the CT scan image
x, y, z = 200, 200, 30

# Resize the saggital and coronal images to make them look better
t_height, t_width = CT_scan.image[:,:,z].shape
transversal_values_resized = CT_scan.image[:,:,z]
transversal_labels_resized = CT_scan.labels[:,:,z]

c_height, c_width = CT_scan.image[:,y,:].shape
coronal_values_resized = utils.resize_image_2d_array(np.int32(CT_scan.image[:,y,:]), 3*c_width, c_height)
coronal_labels_resized = utils.resize_image_2d_array(np.int32(CT_scan.labels[:,y,:]), 3*c_width, c_height)

s_height, s_width = CT_scan.image[x,:,:].shape
saggital_values_resized = utils.resize_image_2d_array(np.int32(CT_scan.image[x,:,:]), 3*s_width, s_height)
saggital_labels_resized = utils.resize_image_2d_array(np.int32(CT_scan.labels[x,:,:]), 3*s_width, s_height)

fig = plt.figure()
a   = fig.add_subplot(2,3,1)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,2)
plt.imshow(coronal_values_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,3)
plt.imshow(saggital_values_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,4)
plt.imshow(CT_scan.labels[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,5)
plt.imshow(coronal_labels_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,6)
plt.imshow(saggital_labels_resized, cmap = cm.Greys_r)
plt.show()