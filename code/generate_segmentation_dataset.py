from dataset_generation.CTScanImage import CTScanImage
import dataset_generation.utils as utils
import dataset_generation.dataset_functions as df
import os
import re
import numpy as np
import h5py

def generate_segmentation_dataset(segmented_CT_scan, slice_center_coordinates, patch_size=32):
	# ********************************************************************************************
	# 								Generate the segmentation datasets
	# ********************************************************************************************
	dicom_height, dicom_width, number_dicoms = segmented_CT_scan.image.shape
	x_slice, y_slice, z_slice 				 = slice_center_coordinates

	print "=======> Generating the segmentation dataset for fixed z = %i from CT scan %s... <=======" %(z_slice, segmented_CT_scan.name)
	tri_planar_segmentation_dataset_fixed_z = np.zeros((segmented_CT_scan.image[:,:,0].size, 6, patch_size, patch_size))
	for x in range(dicom_height):
		utils.drawProgressBar(float(x)/(dicom_height-1), 100)
		for y in range(dicom_width):
			tri_planar_segmentation_dataset_fixed_z[y + dicom_width*x, :, :, :] = df.generate_example_inputs((x,y,z_slice), segmented_CT_scan, patch_size)

	print "=======> Generating the segmentation dataset for fixed y = %i from CT scan %s... <=======" %(y_slice, segmented_CT_scan.name)
	tri_planar_segmentation_dataset_fixed_y = np.zeros((segmented_CT_scan.image[:,0,:].size, 6, patch_size, patch_size))
	for x in range(dicom_height):
		utils.drawProgressBar(float(x)/(dicom_height-1), 100)
		for z in range(number_dicoms):
			tri_planar_segmentation_dataset_fixed_y[z + number_dicoms*x, :, :, :] = df.generate_example_inputs((x,y_slice,z), segmented_CT_scan, patch_size)

	print "=======> Generating the segmentation dataset for fixed x = %i from CT scan %s... <=======" %(x_slice, segmented_CT_scan.name)
	tri_planar_segmentation_dataset_fixed_x = np.zeros((segmented_CT_scan.image[0,:,:].size, 6, patch_size, patch_size))
	for y in range(dicom_width):
		utils.drawProgressBar(float(y)/(dicom_width-1), 100)
		for z in range(number_dicoms):
			tri_planar_segmentation_dataset_fixed_x[z + number_dicoms*y, :, :, :] = df.generate_example_inputs((x_slice,y,z), segmented_CT_scan, patch_size)

	return tri_planar_segmentation_dataset_fixed_x, tri_planar_segmentation_dataset_fixed_y, tri_planar_segmentation_dataset_fixed_z


def save_segmentation_dataset(segmentation_dataset_path, segmented_CT_scan, slice_center_coordinates, dataset_fixed_x, dataset_fixed_y, dataset_fixed_z):
	# ********************************************************************************************
	# 								Saving the segmentation datasets
	# ********************************************************************************************
	x_slice, y_slice, z_slice = slice_center_coordinates
	f = h5py.File(segmentation_dataset_path, "w")

	print "=======> Saving the segmentation datasets for fixed z in %s <=======" %segmentation_dataset_path
	segmentation_dataset_fixed_z 				  = f.create_dataset("segmentation_dataset_fixed_z", dataset_fixed_z.shape, dtype="uint32")
	segmentation_dataset_fixed_z[...] 			  = dataset_fixed_z

	labels_fixed_z 		= f.create_dataset("labels_fixed_z", segmented_CT_scan.labels[:,:,z_slice].shape, dtype="uint32")
	labels_fixed_z[...] = segmented_CT_scan.labels[:,:,z_slice]

	values_fixed_z 		= f.create_dataset("values_fixed_z", segmented_CT_scan.image[:,:,z_slice].shape, dtype="uint32")
	values_fixed_z[...] = segmented_CT_scan.image[:,:,z_slice]

	print "=======> Saving the segmentation datasets for fixed y in %s <=======" %segmentation_dataset_path
	segmentation_dataset_fixed_y 				  = f.create_dataset("segmentation_dataset_fixed_y", dataset_fixed_y.shape, dtype="uint32")
	segmentation_dataset_fixed_y[...] 			  = dataset_fixed_y

	labels_fixed_y 		= f.create_dataset("labels_fixed_y", segmented_CT_scan.labels[:,y_slice,:].shape, dtype="uint32")
	labels_fixed_y[...] = segmented_CT_scan.labels[:,y_slice,:]

	values_fixed_y 		= f.create_dataset("values_fixed_y", segmented_CT_scan.image[:,y_slice,:].shape, dtype="uint32")
	values_fixed_y[...] = segmented_CT_scan.image[:,y_slice,:]

	print "=======> Saving the segmentation datasets for fixed x in %s <=======" %segmentation_dataset_path
	segmentation_dataset_fixed_x 				  = f.create_dataset("segmentation_dataset_fixed_x", dataset_fixed_x.shape, dtype="uint32")
	segmentation_dataset_fixed_x[...] 			  = dataset_fixed_x

	labels_fixed_x 		= f.create_dataset("labels_fixed_x", segmented_CT_scan.labels[x_slice,:,:].shape, dtype="uint32")
	labels_fixed_x[...] = segmented_CT_scan.labels[x_slice,:,:]

	values_fixed_x 		= f.create_dataset("values_fixed_x", segmented_CT_scan.image[x_slice,:,:].shape, dtype="uint32")
	values_fixed_x[...] = segmented_CT_scan.image[x_slice,:,:]

	f.close()

if __name__ == "__main__":
	# ********************************************************************************************
	# 										Setting parameters
	# ********************************************************************************************
	data_directory 				= "../ct_atrium/testing"
	dataset_directory 			= "../datasets/"
	patch_size 					= 32

	CT_scan_parameters_template = {
					"CT_scan_path_template" : os.path.join(data_directory, "CTScan_name"),
					"NRRD_path_template"    : os.path.join(data_directory, "CTScan_name/CTScan_name.nrrd"),
					"DICOM_directory"		: os.path.join(data_directory, "CTScan_name/DICOMS"),
					"DICOM_path_template"   : os.path.join(data_directory, "CTScan_name/DICOMS/DICOM_name"),
					"CT_directory_pattern"  : re.compile("[0-9]{8}")
				}

	testing_CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
	slice_center_coordinates = (230, 230, 30)

	for testing_CT_scan_name in testing_CT_scan_names:
		segmented_CT_scan	 		= CTScanImage(testing_CT_scan_name, CT_scan_parameters_template)
		segmentation_filename 		= "segmentation_datasets_%s.hdf5"%testing_CT_scan_name
		segmentation_dataset_path 	= os.path.join(dataset_directory, segmentation_filename)

		if os.path.isfile(segmentation_dataset_path) == False:
			print segmentation_dataset_path
			tri_planar_segmentation_dataset_fixed_x, tri_planar_segmentation_dataset_fixed_y, tri_planar_segmentation_dataset_fixed_z = generate_segmentation_dataset(segmented_CT_scan, slice_center_coordinates)
			save_segmentation_dataset(
					segmentation_dataset_path, 
					segmented_CT_scan, 
					slice_center_coordinates, 
					tri_planar_segmentation_dataset_fixed_x, 
					tri_planar_segmentation_dataset_fixed_y, 
					tri_planar_segmentation_dataset_fixed_z
				)















