import dataset_generation.utils as utils
import h5py
import numpy as np
import os
import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

experiment_directory 		= "../experimental_results/test_experiment/1"
predicted_labels_path 		= os.path.join(experiment_directory, "predicted_labels.hdf5")
segmentation_dataset_path 	= "../datasets/segmentation_datasets.hdf5"

#******************************************************************************
#							Mask and label plots
#******************************************************************************

# import the label datasets from process_predicted_labels.py 
f_predicted = h5py.File(predicted_labels_path, "r")
predicted_labels_fixed_z = np.array(f_predicted["predicted_labels_fixed_z"])
predicted_labels_fixed_y = np.array(f_predicted["predicted_labels_fixed_y"])
predicted_labels_fixed_x = np.array(f_predicted["predicted_labels_fixed_x"])

# import the true labels
f_segmentation = h5py.File(segmentation_dataset_path, "r")
true_labels_fixed_z = np.array(f_segmentation["labels_fixed_z"]) - 1
true_labels_fixed_y = np.array(f_segmentation["labels_fixed_y"]) - 1
true_labels_fixed_x = np.array(f_segmentation["labels_fixed_x"]) - 1

true_values_fixed_z = np.array(f_segmentation["values_fixed_z"])
true_values_fixed_y = np.array(f_segmentation["values_fixed_y"])
true_values_fixed_x = np.array(f_segmentation["values_fixed_x"])

f_segmentation.close()
f_predicted.close()

# get the masks
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

	return np.array(Image.blend(image, mask, 0.4))

print np.where(predicted_labels_fixed_x == 0)
print np.where(true_labels_fixed_x == 0)

mask_fixed_x = get_mask(predicted_labels_fixed_x, true_labels_fixed_x, true_values_fixed_x)
mask_fixed_y = get_mask(predicted_labels_fixed_y, true_labels_fixed_y, true_values_fixed_y)
mask_fixed_z = get_mask(predicted_labels_fixed_z, true_labels_fixed_z, true_values_fixed_z)

# Resize the saggital and coronal images to make them look better
c_height, c_width = predicted_labels_fixed_x.shape
mask_fixed_y_resized 				= utils.resize_image_2d_array(mask_fixed_y, 4*c_width, c_height)
predicted_labels_fixed_y_resized 	= utils.resize_image_2d_array(np.int32(predicted_labels_fixed_y), 4*c_width, c_height)
true_labels_fixed_y_resized		 	= utils.resize_image_2d_array(np.int32(true_labels_fixed_y), 4*c_width, c_height)

s_height, s_width = predicted_labels_fixed_x.shape
mask_fixed_x_resized 				= utils.resize_image_2d_array(mask_fixed_x, 4*s_width, s_height)
predicted_labels_fixed_x_resized 	= utils.resize_image_2d_array(np.int32(predicted_labels_fixed_x), 4*s_width, s_height)
true_labels_fixed_x_resized		 	= utils.resize_image_2d_array(np.int32(true_labels_fixed_x), 4*c_width, c_height)

fig = plt.figure()
a   = fig.add_subplot(3,3,1)
plt.imshow(predicted_labels_fixed_z, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,4)
plt.imshow(true_labels_fixed_z, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,2)
plt.imshow(predicted_labels_fixed_y_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,5)
plt.imshow(true_labels_fixed_y_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,3)
plt.imshow(predicted_labels_fixed_x_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,6)
plt.imshow(true_labels_fixed_x_resized, cmap = cm.Greys_r)
a   = fig.add_subplot(3,3,7)
plt.imshow(mask_fixed_z)
a   = fig.add_subplot(3,3,8)
plt.imshow(mask_fixed_y_resized)
a   = fig.add_subplot(3,3,9)
plt.imshow(mask_fixed_x_resized)
plt.show()

#******************************************************************************
#							Dice coefficient plots
#******************************************************************************
def read_integers(filename):
    with open(filename) as f:
    	f.readline()
        return [float(x.strip()) for x in f]

training_filename = os.path.join(experiment_directory, "train.log")
training_dice_coefficients = read_integers(training_filename)

testing_filename = os.path.join(experiment_directory, "test.log")
testing_dice_coefficients = read_integers(testing_filename)

x = range(len(testing_dice_coefficients))
plt.plot(x, testing_dice_coefficients)
plt.plot(x, training_dice_coefficients)
plt.legend(['testing', 'training'], loc='lower right')
plt.show()


