from dataset_generation.CTScanImage import CTScanImage
from dataset_generation.DatasetGenerator import DatasetGenerator
import time
import re
import numpy as np
import os


n_examples_per_CT_scan_per_label 	= (200, 200, 400)
sampling_type						= "With_Atrium_Box"
patch_size 							= 32
z 									= 30

# Generate a test dataset from a single DICOM image from a single CT scan
data_directory = "../ct_atrium/testing/"

CT_scan_parameters_template = {
		"CT_scan_path_template" : data_directory + "CTScan_name",
		"NRRD_path_template"    : data_directory + "CTScan_name/CTScan_name.nrrd",
		"DICOM_directory"		: data_directory + "CTScan_name/DICOMS",
		"DICOM_path_template"   : data_directory + "CTScan_name/DICOMS/DICOM_name",
		"CT_directory_pattern"  : re.compile("[0-9]{8}")
		}

CT_scan_names = [directory for directory in os.listdir(data_directory) if CT_scan_parameters_template["CT_directory_pattern"].match(directory)]
CT_scan_name  = CT_scan_names[0]

CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template)
dataset_generator 	= DatasetGenerator(CT_scan, patch_size)

start_time = time.clock()
generated_dataset, generated_labels = dataset_generator.generate_random_dataset(n_examples_per_CT_scan_per_label, sampling_type, multithreaded=False)
total_time = time.clock() - start_time
print "Without multithreading, it takes %f seconds to generate %i examples"%(total_time, sum(n_examples_per_CT_scan_per_label)) 


CT_scan_name  		= CT_scan_names[1]
CT_scan 			= CTScanImage(CT_scan_name, CT_scan_parameters_template)
dataset_generator 	= DatasetGenerator(CT_scan, patch_size)

start_time = time.clock()
generated_dataset, generated_labels = dataset_generator.generate_random_dataset(n_examples_per_CT_scan_per_label, sampling_type, dicom_index=z, multithreaded=True)
total_time = time.clock() - start_time
print "With multithreading, it takes %f seconds to generate %i examples"%(total_time, sum(n_examples_per_CT_scan_per_label))