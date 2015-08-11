from dataset_generation.CTScanImage import CTScanImage
from dataset_generation.DatasetGenerator import DatasetGenerator
import dataset_generation.utils as utils
import os
import re
import numpy as np
import h5py

# ********************************************************************************************
# 										Setting parameters
# ********************************************************************************************
data_directory 				= "../ct_atrium/"
dataset_directory 			= "../datasets/"
segmentation_filename 		= "segmentation_datasets.hdf5"
segmentation_dataset_path 	= os.path.join(dataset_directory, segmentation_filename)

patch_size 					= 32
testing_CT_scans_directory	= os.path.join(data_directory, "testing")

CT_scan_parameters_template = {
				"CT_scan_path_template" : os.path.join(testing_CT_scans_directory, "CTScan_name"),
				"NRRD_path_template"    : os.path.join(testing_CT_scans_directory, "CTScan_name/CTScan_name.nrrd"),
				"DICOM_directory"		: os.path.join(testing_CT_scans_directory, "CTScan_name/DICOMS"),
				"DICOM_path_template"   : os.path.join(testing_CT_scans_directory, "CTScan_name/DICOMS/DICOM_name"),
				"CT_directory_pattern"  : re.compile("[0-9]{8}")
			}

testing_CT_scan_name		= [directory for directory in os.listdir(testing_CT_scans_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)][0]

segmented_CT_scan 		 				 = CTScanImage(testing_CT_scan_name, CT_scan_parameters_template)
dataset_generator 						 = DatasetGenerator(segmented_CT_scan, patch_size)

# ********************************************************************************************
# 								Generate the segmentation datasets
# ********************************************************************************************
dicom_height, dicom_width, number_dicoms = segmented_CT_scan.image.shape
x_slice, y_slice, z_slice 				 = (250, 250, 30)

print "=======> Generating the segmentation dataset for fixed z = %i from CT scan %s... <=======" %(z_slice, segmented_CT_scan.name)
tri_planar_segmentation_dataset_fixed_z = np.zeros((segmented_CT_scan.image[:,:,0].size, 6, patch_size, patch_size))
for x in range(dicom_height):
	utils.drawProgressBar(float(x)/(dicom_height-1), 100)
	for y in range(dicom_width):
		tri_planar_segmentation_dataset_fixed_z[y + dicom_width*x, :, :, :] = dataset_generator.generate_example_inputs((x,y,z_slice))

print "=======> Generating the segmentation dataset for fixed y = %i from CT scan %s... <=======" %(y_slice, segmented_CT_scan.name)
tri_planar_segmentation_dataset_fixed_y = np.zeros((segmented_CT_scan.image[:,0,:].size, 6, patch_size, patch_size))
for x in range(dicom_height):
	utils.drawProgressBar(float(x)/(dicom_height-1), 100)
	for z in range(number_dicoms):
		tri_planar_segmentation_dataset_fixed_y[z + number_dicoms*x, :, :, :] = dataset_generator.generate_example_inputs((x,y_slice,z))

print "=======> Generating the segmentation dataset for fixed x = %i from CT scan %s... <=======" %(x_slice, segmented_CT_scan.name)
tri_planar_segmentation_dataset_fixed_x = np.zeros((segmented_CT_scan.image[0,:,:].size, 6, patch_size, patch_size))
for y in range(dicom_width):
	utils.drawProgressBar(float(y)/(dicom_width-1), 100)
	for z in range(number_dicoms):
		tri_planar_segmentation_dataset_fixed_x[z + number_dicoms*y, :, :, :] = dataset_generator.generate_example_inputs((x_slice,y,z))

# ********************************************************************************************
# 								Saving the segmentation datasets
# ********************************************************************************************
f = h5py.File(segmentation_dataset_path, "w")

print "=======> Saving the segmentation datasets for fixed z in %s <=======" %segmentation_dataset_path
segmentation_dataset_fixed_z 				  = f.create_dataset("segmentation_dataset_fixed_z", tri_planar_segmentation_dataset_fixed_z.shape, dtype="uint32")
segmentation_dataset_fixed_z[...] 			  = tri_planar_segmentation_dataset_fixed_z

labels_fixed_z 		= f.create_dataset("labels_fixed_z", segmented_CT_scan.labels[:,:,z_slice].shape, dtype="uint32")
labels_fixed_z[...] = segmented_CT_scan.labels[:,:,z_slice]

values_fixed_z 		= f.create_dataset("values_fixed_z", segmented_CT_scan.image[:,:,z_slice].shape, dtype="uint32")
values_fixed_z[...] = segmented_CT_scan.image[:,:,z_slice]

print "=======> Saving the segmentation datasets for fixed y in %s <=======" %segmentation_dataset_path
segmentation_dataset_fixed_y 				  = f.create_dataset("segmentation_dataset_fixed_y", tri_planar_segmentation_dataset_fixed_y.shape, dtype="uint32")
segmentation_dataset_fixed_y[...] 			  = tri_planar_segmentation_dataset_fixed_y

labels_fixed_y 		= f.create_dataset("labels_fixed_y", segmented_CT_scan.labels[:,y_slice,:].shape, dtype="uint32")
labels_fixed_y[...] = segmented_CT_scan.labels[:,y_slice,:]

values_fixed_y 		= f.create_dataset("values_fixed_y", segmented_CT_scan.image[:,y_slice,:].shape, dtype="uint32")
values_fixed_y[...] = segmented_CT_scan.image[:,y_slice,:]

print "=======> Saving the segmentation datasets for fixed x in %s <=======" %segmentation_dataset_path
segmentation_dataset_fixed_x 				  = f.create_dataset("segmentation_dataset_fixed_x", tri_planar_segmentation_dataset_fixed_x.shape, dtype="uint32")
segmentation_dataset_fixed_x[...] 			  = tri_planar_segmentation_dataset_fixed_x

labels_fixed_x 		= f.create_dataset("labels_fixed_x", segmented_CT_scan.labels[x_slice,:,:].shape, dtype="uint32")
labels_fixed_x[...] = segmented_CT_scan.labels[x_slice,:,:]

values_fixed_x 		= f.create_dataset("values_fixed_x", segmented_CT_scan.image[x_slice,:,:].shape, dtype="uint32")
values_fixed_x[...] = segmented_CT_scan.image[x_slice,:,:]

f.close()