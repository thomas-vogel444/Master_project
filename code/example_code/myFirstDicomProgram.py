# From the tutorial at https://pyscience.wordpress.com/2014/09/08/dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/

import dicom
import os
import numpy
from matplotlib import pyplot, cm
import re

PathDicom = "../../Master_data/12011501/DICOMS"
a = re.compile("^000[0-9]{5}")

# Retrieves the files names of the DICOM images from the first dataset
listFilesDicom = []
for root, directory, files in os.walk(PathDicom):
	print root
	print directory
	for myfile in files:
		if a.match(myfile):
			listFilesDicom.append(os.path.join(PathDicom, myfile))

ref = dicom.read_file(listFilesDicom[0])

ConstPixelDims = (int(ref.Rows), int(ref.Columns), len(listFilesDicom))

# Load spacing values (in mm)
ConstPixelSpacing = (float(ref.PixelSpacing[0]), float(ref.PixelSpacing[1]), float(ref.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

# The array is sized based on 'ConstPixelDims'
ArrayDicom = numpy.zeros(ConstPixelDims, dtype=ref.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in listFilesDicom:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, listFilesDicom.index(filenameDCM)] = ds.pixel_array  

# Plot a slice of the cat scan
pyplot.figure(dpi=300)
pyplot.axes().set_aspect('equal', 'datalim')
pyplot.set_cmap(pyplot.gray())
pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, 29]))
pyplot.show()