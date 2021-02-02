import os
import re
from skimage.transform import resize
from matplotlib import pyplot
import numpy as np
import cv2

def pixalate_image(image, scale_percent = 40):

    w,h,_ = image.shape
    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # scale back to original size
    dim = (w,h)
    low_res_image = cv2.resize(small_image, dim, interpolation =  cv2.INTER_AREA)
    return low_res_image

## Progress bar is to be added
def Data_Preprocessing(Datapath , Preprocessed_Data_Path):
    
    for root, dirnames, filenames in os.walk(Datapath): # generate the files names
        for filename in filenames:
            if re.search("\.(jpg|jpeg|JPEG|png|bmp|tiff|tif)$", filename):
                
                filepath = os.path.join(root, filename)
                image = pyplot.imread(filepath) # read the image file and save into an array
                if len(image.shape) > 2:
                    # Resize the image so that every image is the same size
                    HighRes = resize(image, (256, 256))
                    # Add this image to the high res dataset
                    # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
                    LowRes = pixalate_image(HighRes)
                    np.save(os.path.join(Preprocessed_Data_Path, filename + '.npy'), HighRes)
                    np.save(os.path.join(Preprocessed_Data_Path, filename + '.npy'), LowRes)  
    print('Done ... ')      
                    
