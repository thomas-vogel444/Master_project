import os
import re
import numpy as np
import utils
import dicom
import nrrd

class CTScanImage:
	""" 
		Class representing the CT scan.
	"""
	def __init__(self, name, parameters_template, xy_padding=0, z_padding=0):
		self.name 				 	  	= name
		self.parameters 		 	  	= self.get_parameters(parameters_template)
		self.dicoms 			 	  	= self.get_DICOM_names()
		self.labels, self.NRRD_header 	= self.get_NRRD_image()
		self.image 					  	= self.get_CT_scan_array()
		self.labels_with_atrium_box 	= self.get_labels_with_atrium_box(xy_padding, z_padding)

	def get_label(self, coordinates):
		"""
			Returns the label for a given set of coordinates. 
		"""
		x, y, z = coordinates
		return self.labels[x, y, z]

	def sample_CT_scan_indices(self, sampling_type, n, target_label=None, z=None):
		"""
			Sample where appropriate, either randomly, with or without an atrium box.
		"""
		if sampling_type == "Random":
			indices_3d = utils.random_3d_indices(np.zeros(self.labels.shape), n, 0, z)
		elif sampling_type == "With_Atrium_Box":
			indices_3d = utils.random_3d_indices(self.labels_with_atrium_box, n, target_label, z)
		elif sampling_type == "Without_Atrium_Box":
			indices_3d = utils.random_3d_indices(self.labels, n, target_label, z)

		return indices_3d

	def get_parameters(self, data_parameters_template):
		return {
			"CT_scan_path" 			: data_parameters_template["CT_scan_path_template"].replace("CTScan_name", self.name),
			"NRRD_path"    			: data_parameters_template["NRRD_path_template"].replace("CTScan_name", self.name),
			"DICOM_directory"		: data_parameters_template["DICOM_directory"].replace("CTScan_name", self.name),
			"DICOM_path_template"   : data_parameters_template["DICOM_path_template"].replace("CTScan_name", self.name),
			"CT_directory_pattern"  : data_parameters_template["CT_directory_pattern"]
		}

	def get_DICOM_names(self):
		"""
			Get the DICOM file names following a given pattern for a given CT scan, i.e. a name with 8 digits in it.
		"""
		DICOM_names = []
		for root, directory, files in os.walk(self.parameters["DICOM_directory"]):
			DICOM_names = [myfile for myfile in files if self.parameters["CT_directory_pattern"].match(myfile)]

		return DICOM_names

	def get_NRRD_image(self):
		"""
			Wrapper around nrrd.read so as to return the transpose of the array.
		"""
		CT_scan_labels, CT_scan_nrrd_header = nrrd.read(self.parameters["NRRD_path"])
		CT_scan_labels = np.transpose(CT_scan_labels, (1,0,2)) + 1
		return CT_scan_labels, CT_scan_nrrd_header

	def get_CT_scan_array(self):	
		"""
			Extract the CT scan image from the DICOM files.
		"""
		# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
		CT_scan_array = np.zeros(self.NRRD_header["sizes"], dtype="float32")
		for dicom_filename in self.dicoms:
		    # read the file
		    dicom_file_path = self.parameters["DICOM_path_template"].replace("DICOM_name", dicom_filename)
		    ds = dicom.read_file(dicom_file_path)

		    # CT_scan_array[:, :, self.dicoms.index(dicom_filename)] = np.array((np.round(ds.pixel_array/float(ds.pixel_array.max())*255)), dtype=np.uint8)
		    CT_scan_array[:, :, self.dicoms.index(dicom_filename)] = ds.pixel_array

		CT_scan_array = CT_scan_array/np.float32(CT_scan_array.max())
		return CT_scan_array

	def get_labels_with_atrium_box(self, xy_padding, z_padding):
		"""
			Returns a modified labels 3D matrix where the labels denote:
				Non-Atrium outside boundary	: 1
				Non_Atrium inside boundary 	: 2
				Atrium 						: 3
		"""
		dimensions 		= self.labels.shape
		atrium_indices 	= np.where(self.labels == 2)

		x_min = np.max(((np.min(atrium_indices[0]) - xy_padding), 0))
		x_max = np.min(((np.max(atrium_indices[0]) + xy_padding), dimensions[0]))
		y_min = np.max(((np.min(atrium_indices[1]) - xy_padding), 0))
		y_max = np.min(((np.max(atrium_indices[1]) + xy_padding), dimensions[1]))
		z_min = np.max(((np.min(atrium_indices[2]) - z_padding), 0))
		z_max = np.min(((np.max(atrium_indices[2]) + z_padding), dimensions[2]))

		boxed_labels = np.matrix.copy(self.labels)
		boxed_labels[x_min:x_max, y_min:y_max, z_min:z_max] += 1
		return boxed_labels















