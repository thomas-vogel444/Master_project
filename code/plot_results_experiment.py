import process_results.results_processing_function as rf
import os


if __name__ == "__main__":
	# experiment_base_name 	= "varying_datasets"
	# experiment_base_name 	= "varying_number_of_convolutional_layers"
	# experiment_base_name 	= "varying_number_of_connected_layers"
	# experiment_base_name	= "varying_number_of_feature_maps"
	# experiment_base_name  = "varying_number_of_connected_hidden_units"
	# experiment_base_name  = "varying_activation_functions"
	# experiment_base_name  = "varying_pooling_functions"
	# experiment_base_name  = "varying_learning_rate"
	experiment_base_name  = "varying_momentum"
	# experiment_base_name  = "varying_dataset_size"

	experiment_base_directory 	= os.path.join("../experimental_results", experiment_base_name)
	segmentation_dataset_path 	= "../datasets/segmentation_datasets.hdf5"
	predicted_dataset_name		= "predicted_labels.hdf5"
	experiment_names 			= rf.get_experiment_names(experiment_base_directory)
	save = False

	# Plotting the masks
	save_filename 	= "mask_results.png"
	masks = rf.get_experiment_set_mask_images(experiment_base_directory, segmentation_dataset_path, predicted_dataset_name)
	# rf.plot_comparative_masks(masks, experiment_names, experiment_base_directory, save_filename, save)

	# Plotting the dice coefficients
	testing_dice_coefficients 	= rf.get_dice_coefficients(experiment_base_directory, type="test")
	training_dice_coefficients 	= rf.get_dice_coefficients(experiment_base_directory, type="train")
	# rf.plot_dice_coefficients(testing_dice_coefficients, experiment_names, experiment_base_directory)
	# rf.plot_dice_coefficients(training_dice_coefficients, experiment_names, experiment_base_directory)

	#********************************************************************************
	# 							More segmentation results
	#********************************************************************************
	experiment_path  = "../experimental_results/varying_momentum/0_5"

	predicted_dataset_names = [	"predicted_labels_segmentation_datasets_14012303.hdf5",
								"predicted_labels_segmentation_datasets_14022801.hdf5",
								"predicted_labels_segmentation_datasets_14031001.hdf5",
								"predicted_labels_segmentation_datasets_14031201.hdf5",
								"predicted_labels_segmentation_datasets_14040204.hdf5",
								"predicted_labels_segmentation_datasets_14051403.hdf5",
								"predicted_labels_segmentation_datasets_14051404.hdf5"]

	segmentation_dataset_paths = [	"../datasets/segmentation_datasets_14012303.hdf5",
									"../datasets/segmentation_datasets_14022801.hdf5",
									"../datasets/segmentation_datasets_14031001.hdf5",
									"../datasets/segmentation_datasets_14031201.hdf5",
									"../datasets/segmentation_datasets_14040204.hdf5",
									"../datasets/segmentation_datasets_14051403.hdf5",
									"../datasets/segmentation_datasets_14051404.hdf5"]

	CT_scan_name = ["14012303","14022801","14031001","14031201","14040204","14051403","14051404"]

	for predicted_dataset_name, segmentation_dataset_path, CT_scan_name in zip(predicted_dataset_names, segmentation_dataset_paths, CT_scan_names):
		predicted_labels_path 	= os.path.join(experiment_path, predicted_dataset_name)
		save_filename			= "mask_results_%s.png"%CT_scan_name

		masks = rf.get_experiment_mask_images(segmentation_dataset_path, predicted_labels_path)
		rf.plot_triplanar_masks(masks, experiment_base_directory, save_filename=save_filename, save=False)


