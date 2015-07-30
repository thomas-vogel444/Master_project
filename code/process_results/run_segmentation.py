import os
from Segmentator import Segmentator


model_directory = "../../experimental_results/varying_learning_rate/0"
segmentation_dataset_directory = "../../datasets/segmentation_datasets.hdf5"

segmentator = Segmentator(model_directory, segmentation_dataset_directory)
segmentator.segment()