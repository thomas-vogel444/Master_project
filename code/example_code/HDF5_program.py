# An HDF5 file is a container for two kinds of objects: datasets, which are array-like collections 
# of data, and groups, which are folder-like containers that hold datasets and other groups. 
# The most fundamental thing to remember when using h5py is:

# Groups work like dictionaries, and datasets work like NumPy arrays

import h5py
import numpy as np

# Starting point is a File object
f = h5py.File("mytestfile.hdf5", "w")

#*************************************************
#						Datasets
#*************************************************

# datasets work like NumPy arrays
dset = f.create_dataset("mydataset", (100,), dtype='i')
print dset.shape
print dset.dtype

dset[...] = np.arange(100)    # The dataset can be manipulated just like a numpy array. 
print dset[0]
print dset[10]
print dset[0:100:10]		  # slicing works exactly the same way

#*************************************************
#						Groups
#*************************************************
	# Groups work like dictionaries
# Think about groups and datasets the same as you would think about files and folders.
# each have a name, and it all starts with the File object we created which is like the
# root group/folder. From there you can create subgroups and datasets easily.
grp = f.create_group("subgroup")
dset2 = grp.create_dataset("another_dataset", (50,), dtype="f")	# create datasets within groups
dset3 = f.create_dataset("subgroup/third_data_set", (10,), dtype="f")    # create a dataset stratight from the File object

# Treat the file object like a dictionary
for name in f:
	print name

"mydataset" in f
"something_else" in f

#*************************************************
#					 Attributes
#*************************************************
# Attributes are metadata attached to groups and datasets which you can play with. You can access it through the
# attrs proxy object.
dset2.attrs['temperature'] = 99.5
print dset2.attrs['temperature']

'temperature' in dset2.attrs


























