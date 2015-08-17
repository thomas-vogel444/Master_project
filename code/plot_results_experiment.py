import dataset_generation.utils as utils
import h5py
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_experiments_paths(experiment_base_directory):
	experiment_names = [directory_name for directory_name in os.listdir(experiment_base_directory)]
	return [os.path.join(experiment_base_directory, experiment_name) for experiment_name in experiment_names
							if os.path.isdir(os.path.join(experiment_base_directory, experiment_name))]

def get_mask_images(experiment_base_directory, segmentation_dataset_path):
	experiment_paths = get_experiments_paths(experiment_base_directory)
	
	masks = {}
	
	for experiment_path in experiment_paths:
		experiment_name 		= os.path.basename(experiment_path)
		predicted_labels_path 	= os.path.join(experiment_path, "predicted_labels.hdf5")

		# import the label datasets from process_predicted_labels.py 
		f_predicted = h5py.File(predicted_labels_path, "r")
		predicted_labels_fixed_x = np.array(f_predicted["predicted_labels_fixed_x"])
		predicted_labels_fixed_y = np.array(f_predicted["predicted_labels_fixed_y"])
		predicted_labels_fixed_z = np.array(f_predicted["predicted_labels_fixed_z"])
		f_predicted.close()

		# import the true labels
		f_segmentation = h5py.File(segmentation_dataset_path, "r")
		true_labels_fixed_x = np.array(f_segmentation["labels_fixed_x"]) - 1
		true_labels_fixed_y = np.array(f_segmentation["labels_fixed_y"]) - 1
		true_labels_fixed_z = np.array(f_segmentation["labels_fixed_z"]) - 1

		true_values_fixed_x = np.array(f_segmentation["values_fixed_x"])
		true_values_fixed_y = np.array(f_segmentation["values_fixed_y"])
		true_values_fixed_z = np.array(f_segmentation["values_fixed_z"])
		f_segmentation.close()

		mask_fixed_x = get_resized_mask(get_mask(predicted_labels_fixed_x, true_labels_fixed_x, true_values_fixed_x))
		mask_fixed_y = get_resized_mask(get_mask(predicted_labels_fixed_y, true_labels_fixed_y, true_values_fixed_y))
		mask_fixed_z = get_mask(predicted_labels_fixed_z, true_labels_fixed_z, true_values_fixed_z)

		masks[experiment_name] = [mask_fixed_x, mask_fixed_y, mask_fixed_z]

	return masks

def get_dice_coefficients(experiment_base_directory, type="test"):
	experiment_paths = get_experiments_paths(experiment_base_directory)

	dice_coefficients = {}

	for experiment_path in experiment_paths:
		experiment_name  = os.path.basename(experiment_path)
		filename = os.path.join(experiment_path, "%s.log"%type)

		dice_coefficients[experiment_name] = read_values(filename)
	return dice_coefficients

def read_values(filename):
    with open(filename) as f:
    	f.readline()
        return [float(x.strip()) for x in f]

def get_mask(predicted_labels, true_labels, true_values):
	"""
		Creates a mask given the predicted and true labels and the true values of a slice of a CT scan.
	"""
	x, y = predicted_labels.shape
	rgbLabels = np.zeros((x, y, 3), 'uint8')
	rgbValues = np.zeros((x, y, 3), 'uint8')

	rgbLabels[:,:,1] = np.where((predicted_labels == true_labels) & (true_labels == 1), 255, 0)
	rgbLabels[:,:,2] = np.where((predicted_labels == true_labels) & (true_labels == 0), 255, 0)
	rgbLabels[:,:,0] = np.where(predicted_labels != true_labels, 255, 0)

	rgbValues[:,:,0] = true_values
	rgbValues[:,:,1] = true_values
	rgbValues[:,:,2] = true_values

	mask  = Image.fromarray(rgbLabels)
	image = Image.fromarray(rgbValues)

	return Image.blend(image, mask, 0.4)

def get_resized_mask(mask):
	width, height = mask.size
	return mask.resize((4*width, height))

#******************************************************************************************
#										Masks plot
#******************************************************************************************
# experiment_base_name 	= "varying_datasets"
# experiment_base_name 	= "varying_number_of_convolutional_layers"
# experiment_base_name 	= "varying_number_of_connected_layers"
# experiment_base_name	= "varying_number_of_feature_maps"
experiment_base_name  = "varying_number_of_connected_hidden_units"

experiment_base_directory 	= os.path.join("../experimental_results", experiment_base_name)
segmentation_dataset_path 	= "../datasets/segmentation_datasets.hdf5"
experiment_names 			= [os.path.basename(path) for path in get_experiments_paths(experiment_base_directory)]

if experiment_names[0].isdigit():
	experiment_names = sorted(experiment_names, key=lambda x: int(x))

masks = get_mask_images(experiment_base_directory, segmentation_dataset_path)

fig = plt.figure()
for i_experiment, experiment_name in enumerate(experiment_names):
	experiment_masks = masks[experiment_name]
	print experiment_name
	for i_mask in range(3):
		a   = fig.add_subplot(3,len(masks),(i_experiment + len(masks) * i_mask + 1))
		plt.imshow(np.array(experiment_masks[i_mask]))
		if i_mask == 0:
			a.set_title(experiment_name)
		plt.axis('off')
# plt.show()
plt.savefig(os.path.join(experiment_base_directory, "mask_results.pdf"))

#******************************************************************************************
#									Testing error plot
#******************************************************************************************
testing_dice_coefficients = get_dice_coefficients(experiment_base_directory, type="test")
training_dice_coefficients = get_dice_coefficients(experiment_base_directory, type="train")

fig = plt.figure()
for experiment_name in experiment_names:
	print experiment_name, max(testing_dice_coefficients[experiment_name])
	plt.plot(testing_dice_coefficients[experiment_name])

plt.legend(experiment_names, loc='lower right')
# plt.show()
plt.savefig(os.path.join(experiment_base_directory, "test_dice_coefficient_plots.pdf"))

fig = plt.figure()
for experiment_name in experiment_names:
	print experiment_name, max(training_dice_coefficients[experiment_name])
	plt.plot(training_dice_coefficients[experiment_name])

plt.legend(experiment_names, loc='lower right')
# plt.show()
plt.savefig(os.path.join(experiment_base_directory, "train_dice_coefficient_plots.pdf"))









