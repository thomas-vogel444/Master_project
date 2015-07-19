import os
import re
import numpy as np
import dicom
import nrrd

class CTScanImage:
	""" 
		Class representing the CT scan.
	"""
	name   	    = None
	parameters  = None
	labels 	    = None
	NRRD_header = None
	image       = None

	def __init__(self, name, parameters_template):
		self.name 				 	  = name
		self.parameters 		 	  = self.get_parameters(parameters_template)
		self.dicoms 			 	  = self.get_DICOM_names()
		self.labels, self.NRRD_header = self.get_NRRD_image()
		self.image 					  = self.get_CT_scan_array()

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
		CT_scan_labels = np.transpose(CT_scan_labels, (1,0,2))
		return CT_scan_labels, CT_scan_nrrd_header

	def get_CT_scan_array(self):	
		"""
			Extract the CT scan image from the DICOM files.
		"""
		# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
		CT_scan_array = np.zeros(self.NRRD_header["sizes"], dtype="uint16")
		for dicom_filename in self.dicoms:
		    # read the file
		    dicom_file_path = self.parameters["DICOM_path_template"].replace("DICOM_name", dicom_filename)
		    ds = dicom.read_file(dicom_file_path)
		    # store the raw image data
		    CT_scan_array[:, :, self.dicoms.index(dicom_filename)] = ds.pixel_array

		return CT_scan_array
