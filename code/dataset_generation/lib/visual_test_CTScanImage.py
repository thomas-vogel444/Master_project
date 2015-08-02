from CTScanImage import CTScanImage
import re
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_directory = "../../../ct_atrium/Testing/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
CT_scan_name  = CT_scan_names[0]

CT_scan = CTScanImage(CT_scan_name, CT_scan_parameters_template)

# Plot a bunch of cross sections from the CT scan image
x, y, z = 250, 250, 30

fig = plt.figure()
a   = fig.add_subplot(2,3,1)
plt.imshow(CT_scan.image[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,2)
plt.imshow(CT_scan.image[:,y,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,3)
plt.imshow(CT_scan.image[x,:,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,4)
plt.imshow(CT_scan.labels[:,:,z], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,5)
plt.imshow(CT_scan.labels[:,y,:], cmap = cm.Greys_r)
a   = fig.add_subplot(2,3,6)
plt.imshow(CT_scan.labels[x,:,:], cmap = cm.Greys_r)
plt.show()