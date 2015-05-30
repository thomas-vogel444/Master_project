import dicom
import h5py
import os
import re
import pprint as pp
import numpy 

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
	
#*****************************************************************************************
# For a given dicom file, produce a 4d numpy array Batch Channel Width Height
# with:
#	Batch   : number of training examples, i.e. number of voxels in the image 480 * 480
#	Channel : 3 planes centred at the voxel
#	Width   : width of the image path, i.e. 32
#	Height  : height of the image patch, i.e. 32
#*****************************************************************************************
# Get the path for a given CT scan
CT_scan_names           = ct_scan_dictionary.keys()
CT_scan_name 			= CT_scan_names[0]
CT_scan_dicom_filenames = ct_scan_dictionary[CT_scan_name]

dicom_height  = 480
dicom_width   = 480
number_dicoms = len(CT_scan_dicom_filenames)

# Loop through all the DICOM files for a given CT scan
ArrayDicom = numpy.zeros((dicom_height, dicom_width, number_dicoms), dtype="uint16")
for dicom_filename in CT_scan_dicom_filenames:
    # read the file
    dicom_file_path = DICOM_path.replace("CTScan_name", CT_scan_name).replace("DICOM_name", dicom_filename)
    print "Reading %s" %(dicom_file_path)
    ds = dicom.read_file(dicom_file_path)
    # store the raw image data
    ArrayDicom[:, :, CT_scan_dicom_filenames.index(dicom_filename)] = ds.pixel_array  
















