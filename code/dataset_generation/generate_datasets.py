import dicom
import h5py
import os
import re
import pprint as pp
import numpy as np

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
# For a given dicom file, produce a 4D numpy array Batch Channel Width Height
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

patch_size 	  = 32
dicom_height  = 480
dicom_width   = 480
number_dicoms = len(CT_scan_dicom_filenames)

# Loop through all the DICOM files for a given CT scan and get all the values into a single 3D numpy array
ArrayDicom = np.zeros((dicom_height, dicom_width, number_dicoms), dtype="uint16")
for dicom_filename in CT_scan_dicom_filenames:
    # read the file
    dicom_file_path = DICOM_path.replace("CTScan_name", CT_scan_name).replace("DICOM_name", dicom_filename)
    print "Reading %s" %(dicom_file_path)
    ds = dicom.read_file(dicom_file_path)
    # store the raw image data
    ArrayDicom[:, :, CT_scan_dicom_filenames.index(dicom_filename)] = ds.pixel_array  

# For each voxel, produce 3 32*32 perpendicular patches with it as their centre. 
tri_planar_dataset = np.zeros((ArrayDicom.size, 3, patch_size, patch_size))
x_grid = np.arange(dicom_height)
y_grid = np.arange(dicom_width)
z_grid = np.arange(number_dicoms)


# For an edge voxel
x = dicom_height - 1
y = 20
z = number_dicoms - patch_size/2 + 1
# y = patch_size/2 - 1
# z = patch_size/2 - 1
plane_1 = np.zeros((patch_size, patch_size))
plane_2 = np.zeros((patch_size, patch_size))
plane_3 = np.zeros((patch_size, patch_size))

x_min = 0
x_max = patch_size
y_min = 0
y_max = patch_size
z_min = 0
z_max = patch_size

if x < patch_size/2:
	x_min = patch_size/2 - x

if x > dicom_height - patch_size/2:
	x_max = patch_size/2 + dicom_height - x

if y < patch_size/2:
	y_min = patch_size/2 - y

if y > dicom_width - patch_size/2:
	y_max = patch_size/2 + dicom_width - y

if z < patch_size/2:
	z_min = patch_size/2 - z

if z > number_dicoms - patch_size/2:
	z_max = patch_size/2 + number_dicoms - z

ArrayDicom.fill(1)

plane_1[x_min:x_max, y_min:y_max] = ArrayDicom[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, dicom_height), 
					 np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, dicom_width), 
					 z]

plane_2[x_min:x_max, z_min:z_max] = ArrayDicom[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, dicom_height), 
					 y, 
					 np.maximum(z-patch_size/2, 0):np.minimum(z+patch_size/2, number_dicoms)]

plane_3[y_min:y_max, z_min:z_max] = ArrayDicom[x, 
					 np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, dicom_width), 
					 np.maximum(z-patch_size/2, 0):np.minimum(z+patch_size/2, number_dicoms)]

print plane_1
print plane_1.shape
print plane_2
print plane_2.shape
print plane_3
print plane_3.shape


# #***************************************************************
# from matplotlib import pyplot, cm
# path = "../../ct_atrium/14022803/DICOMS/00010088"


# ref = dicom.read_file(path)

# ConstPixelDims = (int(ref.Rows), int(ref.Columns), number_dicoms)

# # Load spacing values (in mm)
# ConstPixelSpacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]), float(ref.SliceThickness))

# x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
# y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
# z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# pyplot.figure(dpi=300)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.set_cmap(pyplot.gray())
# pyplot.pcolormesh(x, y, np.flipud(ArrayDicom[:, :, 29]))
# pyplot.show()











