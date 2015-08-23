import h5py
import numpy as np
import Image
import sys

def drawProgressBar(percent, bar_length = 20):
    """
        Draws and updates a progress bar.
    """
    sys.stdout.write("\r")
    progress = ""
    for i in range(bar_length):
        if i < int(bar_length * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent*100))
    if percent == 1:
		sys.stdout.write("\n")    	
    sys.stdout.flush()

def save_dataset(dataset_path, dataset, labels=None):
    """ 
        Saves a dataset and labels to a given path.
    """
    f = h5py.File(dataset_path, "w")
    dataset_hdf5        = f.create_dataset("dataset", dataset.shape, dtype="float32")
    dataset_hdf5[...]   = dataset

    if labels != None:
        labels_hdf5         = f.create_dataset("labels", labels.shape, dtype="uint8")
        labels_hdf5[...]    = labels
    f.close()

def random_3d_indices(CT_scan_labels, n, target_label, z=None):
    """
        Randomly selects n data points from the set of points of a given label and returns their indices.
    """
    if z is None:
        indices_3d = np.where(CT_scan_labels == target_label)
    else:
        indices_3d = list(np.where(CT_scan_labels[:,:,z] == target_label))
        indices_3d.append(np.ones(len(indices_3d[0]), dtype=np.int)*z)
    
    indices_1d = np.random.choice(xrange(len(indices_3d[0])), min(n,len(indices_3d[0])), replace=False)

    return np.dstack((  indices_3d[0][indices_1d], 
                        indices_3d[1][indices_1d], 
                        indices_3d[2][indices_1d]))[0]

def padded_square_image_crop(center_coordinates,image_2d, patch_size):
    """
        Generate a single patch of size patch_size*patch_size.
    """
    x, y            = center_coordinates
    height, width   = image_2d.shape
    patch           = np.zeros((patch_size, patch_size))

    x_min = 0
    x_max = patch_size
    y_min = 0
    y_max = patch_size

    if x < patch_size/2:
        x_min = patch_size/2 - x

    if x > height - patch_size/2:
        x_max = patch_size/2 + height - x

    if y < patch_size/2:
        y_min = patch_size/2 - y

    if y > width - patch_size/2:
        y_max = patch_size/2 + width - y

    patch[x_min:x_max, y_min:y_max] = image_2d[np.maximum(x-patch_size/2, 0):np.minimum(x+patch_size/2, height), 
                         np.maximum(y-patch_size/2, 0):np.minimum(y+patch_size/2, width)]

    return patch

def resize_image_2d_array(image_2d_array, height, width):
    """
        resizes a 2d image array to have dimension height, width.
    """
    im = Image.fromarray(image_2d_array)
    out = im.resize((height, width))
    return np.array(out)