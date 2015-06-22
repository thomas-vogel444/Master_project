import dicom
import h5py
import nrrd
import os
import re
import pprint as pp
import numpy as np
import dataset_functions as df

if __name__ == "__main__":
	# Setting Parameters
	data_directory 		  = "../../ct_atrium"
	CT_scan_path_template = "../../ct_atrium/CTScan_name"
	NRRD_path_template    = "../../ct_atrium/CTScan_name/CTScan_name.nrrd"
	DICOM_path_template   = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"
	ct_directory_pattern  = re.compile("[0-9]{8}")
	patch_size    		  = 32

	# Get the names of all the CT scans
	CT_scans = df.get_CT_scan_names(data_directory, ct_directory_pattern)

	# For every CT scan, get a list of their associated DICOM files
	CT_scan_dictionary = df.get_all_DICOMs(CT_scan_path_template, CT_scans)

	dataset_directory = os.path.join(data_directory, "datasets")
	for CT_scan, DICOMs in CT_scan_dictionary.items()[0:1]:
		# For every CT scan, produce a hdf5 file
		datafile_name = "dataset_CTScan_Name.hdf5".replace("CTScan_Name", CT_scan)
		dataset_path  = os.path.join(dataset_directory, datafile_name)
		f 			  = h5py.File(dataset_path, "w")

		# Get the atlas from the NRRD file
		nrrd_path = NRRD_path_template.replace("CTScan_name", CT_scan)
		CT_scan_labels, CT_scan_nrrd_header = nrrd.read(nrrd_path)

		# Extract the 3d image into a numpy array
		print "Extracting the data from the DICOM files for CT scan %s" % CT_scan
		print "CT scan number %i out of %i" %(CT_scan_dictionary.keys().index(CT_scan), len(CT_scan_dictionary.keys()))
		CT_scan_3d_image = df.get_CT_scan_array(CT_scan, DICOMs, CT_scan_nrrd_header["sizes"], DICOM_path_template)

		# Generate 3 perpendicular patches for each data point
		# For each voxel, produce 3 32*32 perpendicular patches with it as their centre. 
		dicom_height, dicom_width, number_dicoms = CT_scan_3d_image.shape

		x_grid = np.arange(dicom_height)
		y_grid = np.arange(dicom_width)
		z_grid = np.arange(number_dicoms)

		tri_planar_dataset = np.zeros((CT_scan_3d_image[:,:,0].size, 3, patch_size, patch_size))

		# Looping over all the dicom files
		for z in z_grid:
			print "Generating patches for dicom file number %i..." %(z)
			for x in y_grid:
				for y in x_grid:
					tri_planar_dataset[y + dicom_width*x, :, :, :] = df.tri_planar_patch_generator(x,y,z,CT_scan_3d_image,patch_size)
			
			# Save the set of patches into a dataset. Each data element is saved in 1 byte of memory as it takes values between 0 and 255.
			patches_dataset_name = "patches_%s_%i"%(CT_scan, z)
			print "Saving dataset of patches %s..." %patches_dataset_name
			dataset 	 = f.create_dataset(patches_dataset_name, tri_planar_dataset.shape, dtype="uint8")
			dataset[...] = tri_planar_dataset

			# Save the labels into a separate dataset, one per DICOM image.
			labels_dataset_name = "labels_%s_%i"%(CT_scan, z)
			print "Saving dataset of labels %s..." %labels_dataset_name
			print "=============>" +  str(tri_planar_dataset.shape[0])
			dataset 	 = f.create_dataset(labels_dataset_name, (tri_planar_dataset.shape[0],), dtype="uint8")
			dataset[...] = np.ravel(np.transpose(CT_scan_labels[:,:,z]))
	
		f.close()
		print "The datasets for %s have been saved into %s" % (CT_scan, dataset_path)

