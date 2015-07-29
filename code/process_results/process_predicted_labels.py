import h5py

# import the label datasets from process_predicted_labels.py 
predicted_labels_path = "predicted_labels.hdf5"
f = h5py.File(predicted_labels_path, "r")
predicted_labels_fixed_z = f["predicted_labels_fixed_z"]
predicted_labels_fixed_y = f["predicted_labels_fixed_y"]
predicted_labels_fixed_x = f["predicted_labels_fixed_x"]
f.close()

# import the image and labels
segmentation_dataset_path = "../../datasets/segmentation_datasets.hdf5"
f = h5py.File(segmentation_dataset_path, "r")
segmentation_dataset_fixed_z = f["segmentation_dataset_fixed_z"]
segmentation_dataset_fixed_y = f["segmentation_dataset_fixed_y"]
segmentation_dataset_fixed_x = f["segmentation_dataset_fixed_x"]
f.close()

# Plot masks using the image library