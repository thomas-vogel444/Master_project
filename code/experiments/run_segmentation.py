import os
from Segmentator import Segmentator


segmentation_code_path	= "../CNN/segment.lua"
model_directory 		= "../../experimental_results/varying_learning_rate/0"
segmentation_file_path 	= "../../datasets/segmentation_datasets.hdf5"

segmentator = Segmentator(segmentation_file_path, segmentation_code_path)
segmentator.segment(model_directory)