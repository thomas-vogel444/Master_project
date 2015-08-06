import numpy as np
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

    return np.dstack((indices_3d[0][indices_1d], 
                                            indices_3d[1][indices_1d], 
                                            indices_3d[2][indices_1d]))[0]