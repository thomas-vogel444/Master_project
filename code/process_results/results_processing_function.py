import sys
sys.path.append("..")
import dataset_generation.utils as utils
import h5py
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_experiment_names(experiment_base_directory):
	experiment_names = [os.path.basename(path) for path in get_experiments_paths(experiment_base_directory)]
	if experiment_names[0].isdigit():
		experiment_names = sorted(experiment_names, key=lambda x: int(x))

	return experiment_names

def get_experiments_paths(experiment_base_directory):
	experiment_names = [directory_name for directory_name in os.listdir(experiment_base_directory)]
	return [os.path.join(experiment_base_directory, experiment_name) for experiment_name in experiment_names
							if os.path.isdir(os.path.join(experiment_base_directory, experiment_name))]

def get_predicted_labels(predicted_labels_path, dataset_name):
	f = h5py.File(predicted_labels_path, "r")
	predicted_labels = np.array(f[dataset_name])
	f.close()
	return predicted_labels

def get_true_labels(segmentation_dataset_path, dataset_name):
	f = h5py.File(segmentation_dataset_path, "r")
	true_labels = np.array(f[dataset_name]) - 1
	f.close()
	return true_labels

def get_true_values(segmentation_dataset_path, dataset_name):
	f 			= h5py.File(segmentation_dataset_path, "r")
	true_values = np.array(f[dataset_name])
	f.close()
	return true_values

def get_experiment_mask_images(segmentation_dataset_path, predicted_labels_path):
	print predicted_labels_path
	print segmentation_dataset_path
	# import the label datasets from process_predicted_labels.py 
	predicted_labels_fixed_x = get_predicted_labels(predicted_labels_path, "predicted_labels_fixed_x")
	predicted_labels_fixed_y = get_predicted_labels(predicted_labels_path, "predicted_labels_fixed_y")
	predicted_labels_fixed_z = get_predicted_labels(predicted_labels_path, "predicted_labels_fixed_z")

	true_labels_fixed_x = get_true_labels(segmentation_dataset_path, "labels_fixed_x")
	true_labels_fixed_y = get_true_labels(segmentation_dataset_path, "labels_fixed_y")
	true_labels_fixed_z = get_true_labels(segmentation_dataset_path, "labels_fixed_z")

	true_values_fixed_x = get_true_values(segmentation_dataset_path, "values_fixed_x")
	true_values_fixed_y = get_true_values(segmentation_dataset_path, "values_fixed_y")
	true_values_fixed_z = get_true_values(segmentation_dataset_path, "values_fixed_z")

	# import the true labels
	mask_fixed_x = get_resized_mask(get_mask(predicted_labels_fixed_x, true_labels_fixed_x, true_values_fixed_x))
	mask_fixed_y = get_resized_mask(get_mask(predicted_labels_fixed_y, true_labels_fixed_y, true_values_fixed_y))
	mask_fixed_z = get_mask(predicted_labels_fixed_z, true_labels_fixed_z, true_values_fixed_z)

	return [mask_fixed_x, mask_fixed_y, mask_fixed_z]

def get_experiment_set_mask_images(experiment_base_directory, segmentation_dataset_path, predicted_dataset_name):
	experiment_paths = get_experiments_paths(experiment_base_directory)
	
	masks = {}
	
	for experiment_path in experiment_paths:
		experiment_name 		= os.path.basename(experiment_path)
		predicted_labels_path 	= os.path.join(experiment_path, predicted_dataset_name)
		print predicted_labels_path

		masks[experiment_name] = get_experiment_mask_images(segmentation_dataset_path, predicted_labels_path)

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
	rgbLabels[:,:,2] = np.where((true_labels == 0), 255, 0)
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

def plot_comparative_masks(masks, experiment_names, experiment_base_directory, save_filename=None, save=False):
	"""
		Plot the transversal slice of the segmentation data for comparison sake...
	"""
	fig = plt.figure()
	for i_experiment, experiment_name in enumerate(experiment_names):
		experiment_masks = masks[experiment_name]
		print experiment_name
		a   = fig.add_subplot(2,np.ceil(len(experiment_names)/2.),i_experiment + 1)
		plt.imshow(np.array(experiment_masks[2]))
		a.set_title(experiment_name)
		plt.axis('off')
	if save == True:
		plt.savefig(os.path.join(experiment_base_directory, save_filename))
	else:
		plt.show()

def plot_triplanar_masks(masks, experiment_base_directory, save_filename=None, save=False):
	"""
		Plot all three slices of the triplanar segmentation.
	"""
	fig = plt.figure()
	for i_mask in range(3):
		a   = fig.add_subplot(1,len(masks),i_mask + 1)
		plt.imshow(np.array(masks[i_mask]))
		plt.axis('off')
	if save == True:
		plt.savefig(os.path.join(experiment_base_directory, save_filename))
	else:
		plt.show()

def plot_dice_coefficients(dice_coefficients, experiment_names, experiment_base_directory, saving_filename=None, save=False):
	fig = plt.figure()
	for experiment_name in experiment_names:
		print experiment_name, max(dice_coefficients[experiment_name])
		plt.plot(dice_coefficients[experiment_name])
		plt.ylim((80,100))

	plt.legend(experiment_names, loc='lower right')

	if save == True:
		plt.savefig(os.path.join(experiment_base_directory, saving_filename))
	else:
		plt.show()