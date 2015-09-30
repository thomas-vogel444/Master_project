import sys
sys.path.append("..")
from CTScanImage import CTScanImage
import utils
import dataset_functions as df
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import re

data_directory = "../../../ct_atrium/training/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_name 	= "14022803" 
x 				= 250
y 				= 200
z 				= 40
patch_size 		= 32

CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template)

patches = df.generate_example_inputs((x,y,z), CT_scan, patch_size)

# Plot 3 patches next to their respective DICOM slice.
fig = plt.figure()
a   = fig.add_subplot(3,4,1)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r)
plt.scatter(y,x)
plt.axis('off')
a   = fig.add_subplot(3,4,2)
plt.imshow(patches[0,:,:], cmap = cm.Greys_r)
plt.axis('off')
a   = fig.add_subplot(3,4,3)
plt.imshow(patches[3,:,:], cmap = cm.Greys_r)
plt.axis('off')
a   = fig.add_subplot(3,4,4)
plt.imshow(CT_scan.labels[:,:,z], cmap = cm.Greys_r)
plt.axis('off')
plt.scatter(y,x)

a   = fig.add_subplot(3,4,5)
plt.imshow(utils.resize_image_2d_array(CT_scan.image[:,y,:], 480/3, 480), cmap = cm.Greys_r)
plt.scatter(z,x)
plt.axis('off')

a   = fig.add_subplot(3,4,6)
plt.imshow(patches[1,:,:], cmap = cm.Greys_r)
plt.axis('off')

a   = fig.add_subplot(3,4,7)
plt.imshow(patches[4,:,:], cmap = cm.Greys_r)
plt.axis('off')

a   = fig.add_subplot(3,4,8)
plt.imshow(utils.resize_image_2d_array(CT_scan.labels[:,y,:], 480/3, 480), cmap = cm.Greys_r)
plt.axis('off')
plt.scatter(z,x)

a   = fig.add_subplot(3,4,9)
plt.imshow(utils.resize_image_2d_array(CT_scan.image[x,:,:], 480/3, 480), cmap = cm.Greys_r)
plt.axis('off')
plt.scatter(z,y)
a   = fig.add_subplot(3,4,10)
plt.imshow(patches[2,:,:], cmap = cm.Greys_r)
plt.axis('off')
a   = fig.add_subplot(3,4,11)
plt.imshow(patches[5,:,:], cmap = cm.Greys_r)
plt.axis('off')
a   = fig.add_subplot(3,4,12)
plt.imshow(utils.resize_image_2d_array(CT_scan.labels[x,:,:], 480/3, 480), cmap = cm.Greys_r)
plt.axis('off')
plt.scatter(z,y)
fig.tight_layout()
plt.show()