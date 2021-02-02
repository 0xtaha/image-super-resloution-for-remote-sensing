import os
import re
from skimage.transform import resize, rescale
from matplotlib import pyplot
import numpy as np
import cv2




def Data_Preprocessing(path , just_load_dataset=False):
    
    for root, dirnames, filenames in os.walk(path): # generate the files names
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff|tif)$", filename):
                
                filepath = os.path.join(root, filename)
                image = pyplot.imread(filepath) # read the image file and save into an array
                if len(image.shape) > 2:
                    # Resize the image so that every image is the same size
                    HighRes = resize(image, (256, 256))
                    # Add this image to the high res dataset
                    # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
                    # define new method of preprocessing
                    #LowRes = 
                    
