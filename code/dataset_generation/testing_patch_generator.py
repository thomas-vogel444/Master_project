import matplotlib.pyplot as plt
import numpy as np
import plotting_functions as pf
import dataset_functions as df
import nrrd

CT_scan 		   = "14022803" 
CT_scan_directory  = "../../ct_atrium/14022803"
CT_scan_dicoms     = df.get_DICOMs(CT_scan_directory)

NRRD_path_template = "../../ct_atrium/CTScan_name/CTScan_name.nrrd"
nrrd_path = NRRD_path_template.replace("CTScan_name", CT_scan)
CT_scan_labels, CT_scan_nrrd_header = nrrd.read(nrrd_path)

DICOM_path_template = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"
CT_scan_image     = df.get_CT_scan_array(CT_scan, CT_scan_dicoms, CT_scan_nrrd_header["sizes"], DICOM_path_template)

# Coordinates of a given voxel
x = 20
y = 20
z = 20

# Get the patches at a given voxel
patch_size = 32
patches = df.tri_planar_patch_generator(x,y,z,CT_scan_image,patch_size)
print CT_scan_image[:,:,z].shape
print patches.shape

# Plot 3 patches next to their respective DICOM slice.
fig = plt.figure()
# a   = fig.add_subplot(1,2,1)
pf.plot_2d_image(CT_scan_image[:,:,z])
pf.plot_2d_image(patches[0,:,:])

pf.plot_2d_image(CT_scan_image[:,y,:])
pf.plot_2d_image(patches[1,:,:])

pf.plot_2d_image(CT_scan_image[x,:,:])
pf.plot_2d_image(patches[2,:,:])

# a   = fig.add_subplot(1,2,2)
# pf.plot_2d_image(image)