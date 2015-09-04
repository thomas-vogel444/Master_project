import numpy as np
import Image

def convert_to_grey_scale(array):
	"""
		Converts a 0-1 scale array to the 0-255 grey sale
	"""
	return np.round(array*255)

def generate_mask(predicted_labels, true_labels, true_values):
	"""
		Creates a mask given the predicted and true labels and the true values of a slice of a CT scan.
	"""
	x, y = predicted_labels.shape
	rgbLabels = np.zeros((x, y, 3), 'uint8')
	rgbValues = np.zeros((x, y, 3), 'uint8')

	rgbLabels[:,:,1] = np.where((predicted_labels == true_labels) & (true_labels == 2), 255, 0)
	rgbLabels[:,:,2] = np.where((true_labels == 1), 255, 0)
	rgbLabels[:,:,0] = np.where(predicted_labels != true_labels, 255, 0)


	rgbValues[:,:,0] = convert_to_grey_scale(true_values)
	rgbValues[:,:,1] = convert_to_grey_scale(true_values)
	rgbValues[:,:,2] = convert_to_grey_scale(true_values)

	mask  = Image.fromarray(rgbLabels)
	image = Image.fromarray(rgbValues)

	return Image.blend(image, mask, 0.25)


class SegmentedCTScan:
	"""
		SegmentedCTScan is responsible for handling all operations one might want to do
		on segmented CT scans such as generating masks or getting classification statistics.
	"""
	def __init__(self, CT_scan, predicted_labels):
		self.CT_scan 				= CT_scan
		self.predicted_labels 		= predicted_labels
		self.classification_statistics 	= self.get_classification_statistics()

	def get_classification_statistics(self):
		classification_statistics = {}
		
		number_of_atrium_voxels 				= len(np.where(self.CT_scan.labels == 2)[0])
		number_of_non_atrium_voxels				= len(np.where(self.CT_scan.labels == 1)[0])
		number_correctly_classified_atrium 		= np.sum(np.where((self.predicted_labels == self.CT_scan.labels) & 
																  (self.CT_scan.labels == 2), 1, 0))
		number_correctly_classified_non_atrium 	= np.sum(np.where((self.predicted_labels == self.CT_scan.labels) & 
																  (self.CT_scan.labels == 1), 1, 0))

		classification_statistics["Sensitivity"] = round(number_correctly_classified_atrium/float(number_of_atrium_voxels), 3)
		classification_statistics["Specificity"] = round(number_correctly_classified_non_atrium/float(number_of_non_atrium_voxels), 3)
		classification_statistics["Dice Coefficient"] = round(	(number_correctly_classified_atrium + 
															number_correctly_classified_non_atrium) /
															float(self.predicted_labels.size), 3)
		return classification_statistics

	def get_mask(self, position, fixed_coordinate):
		"""
			Creates a mask given the predicted and true labels and the true values of a slice of a CT scan.
		"""
		assert fixed_coordinate == "x" or fixed_coordinate == "y" or fixed_coordinate == "z", \
				"fixed coordinates is not 'x', 'y' or 'z': %r"%fixed_coordinate

		if fixed_coordinate == "z":
			mask = generate_mask(self.predicted_labels[:,:,position], 
								 self.CT_scan.labels[:,:,position],
								 self.CT_scan.image[:,:,position])
		elif fixed_coordinate == "y":
			mask = generate_mask(self.predicted_labels[:,position,:], 
								 self.CT_scan.labels[:,position,:],
								 self.CT_scan.image[:,position,:])
		elif fixed_coordinate == "x":
			mask = generate_mask(self.predicted_labels[position,:,:], 
								 self.CT_scan.labels[position,:,:],
								 self.CT_scan.image[position,:,:])

		return mask









