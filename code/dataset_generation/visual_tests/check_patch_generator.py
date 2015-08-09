import sys
sys.path.append("..")
import dataset_functions as df
from CTScanImage import CTScanImage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re
import dataset_functions as df

data_directory = "../../../ct_atrium/training/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_name = "14022803" 

CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template)

# Coordinates of a given voxel
x = 250
y = 200
z = 40

# Get the patches at a given voxel
patch_size = 32
patches = df.generate_patches((x,y,z), CT_scan.image, patch_size)

# Plot 3 patches next to their respective DICOM slice.
fig = plt.figure()
a   = fig.add_subplot(1,4,1)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(y,x)
a   = fig.add_subplot(1,4,2)
plt.imshow(patches[0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,3)
plt.imshow(patches[3,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,4)
plt.imshow(CT_scan.labels[:,:,z], cmap = cm.Greys_r)
plt.scatter(y,x)
plt.show()

fig = plt.figure()
a   = fig.add_subplot(1,4,1)
plt.imshow(CT_scan.image[:,y,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(z,x)
a   = fig.add_subplot(1,4,2)
plt.imshow(patches[1,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,3)
plt.imshow(patches[4,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,4)
plt.imshow(CT_scan.labels[:,y,:], cmap = cm.Greys_r)
plt.scatter(z,x)
plt.show()

fig = plt.figure()
a   = fig.add_subplot(1,4,1)
plt.imshow(CT_scan.image[x,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(z,y)
a   = fig.add_subplot(1,4,2)
plt.imshow(patches[2,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,3)
plt.imshow(patches[5,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,4)
plt.imshow(CT_scan.labels[x,:,:], cmap = cm.Greys_r)
plt.scatter(z,y)
plt.show()