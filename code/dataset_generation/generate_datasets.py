import dicom
import h5py
import os
import re
import pprint as pp

# Get all the CT scan folder names
data_directory = "../../ct_atrium"
ct_scan_path = "../../ct_atrium/CTScan_name"
DICOM_path = "../../ct_atrium/CTScan_name/DICOMS/DICOM_name"

ct_directory_pattern = re.compile("[0-9]{8}")
ListCTScans = [directory for directory in os.listdir(data_directory) if ct_directory_pattern.match(directory)]

#************************************************************
# For each CT scan folder get all the DICOM names
#************************************************************
ct_scan_dictionary = {}
for CT_scan_name in ListCTScans:

	# Get the path of a specific CT scan data directory
	ct_scan_directory = ct_scan_path.replace("CTScan_name", CT_scan_name)

	# Get the DICOM file names into a dicstionay with the CT scan name as value and a list of DICOM image names as values
	dicom_directory = os.path.join(ct_scan_directory, "DICOMS")
	dicom_name_pattern = re.compile("^[0-9]{8}")
	dicom_files = []
	for root, directory, files in os.walk(dicom_directory):
		dicom_files = [myfile for myfile in files if dicom_name_pattern.match(myfile)]

	ct_scan_dictionary[CT_scan_name] = dicom_files
	


