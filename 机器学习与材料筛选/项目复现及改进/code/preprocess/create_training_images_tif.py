import tifffile
import numpy as np
import h5py
import argparse

for i in range(0, 1280):
    name = 'D:/test_MistGPU/training_images_0.6tif/example_'+str(i)+'.tif'
    img = tifffile.imread(name)
    f = h5py.File('D:/test_MistGPU/training_images_0.6'+'/'+'test_'+str(i)+".hdf5", "w")
    f.create_dataset('data', data=img, dtype="i8", compression="gzip")
    f.close()