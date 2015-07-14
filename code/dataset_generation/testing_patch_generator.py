import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import lib.dataset_functions as df

CT_scan 		   = "14022803" 
CT_scan_directory  = "../../ct_atrium/14022803"
CT_scan_dicoms     = df.get_DICOMs(CT_scan_directory)

NRRD_path_template = "../../ct_atrium/CTScan_name/CTScan_name.nrrd"
nrrd_path = NRRD_path_template.replace("CTScan_name", CT_scan)
CT_scan_labels, CT_scan_nrrd_header = df.get_NRRD_array(nrrd_path)

DICOM_path_template = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"
CT_scan_image     = df.get_CT_scan_array(CT_scan, CT_scan_dicoms, CT_scan_nrrd_header["sizes"], DICOM_path_template)

# Coordinates of a given voxel
x = 250
y = 200
z = 40

# Get the patches at a given voxel
patch_size = 32
patches = df.generate_patches(x,y,z,CT_scan_image,patch_size)
np.set_printoptions(threshold=np.nan)

# Plot 3 patches next to their respective DICOM slice.
# print CT_scan_labels[x,y,z]
fig = plt.figure()
a   = fig.add_subplot(1,3,1)
plt.imshow(CT_scan_image[:,:,z], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(y,x)
a   = fig.add_subplot(1,3,2)
plt.imshow(patches[0,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,3,3)
plt.imshow(CT_scan_labels[:,:,z], cmap = cm.Greys_r)
plt.scatter(y,x)
plt.show()

fig = plt.figure()
a   = fig.add_subplot(1,4,1)
plt.imshow(CT_scan_image[:,y,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(z,x)
a   = fig.add_subplot(1,4,2)
plt.imshow(patches[1,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,3)
plt.imshow(patches[4,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,4)
plt.imshow(CT_scan_labels[:,y,:], cmap = cm.Greys_r)
plt.scatter(z,x)
plt.show()

fig = plt.figure()
a   = fig.add_subplot(1,4,1)
plt.imshow(CT_scan_image[x,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
plt.scatter(z,y)
a   = fig.add_subplot(1,4,2)
plt.imshow(patches[2,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,3)
plt.imshow(patches[5,:,:], cmap = cm.Greys_r, vmin = 0, vmax = 500)
a   = fig.add_subplot(1,4,4)
plt.imshow(CT_scan_labels[x,:,:], cmap = cm.Greys_r)
plt.scatter(z,y)
plt.show()