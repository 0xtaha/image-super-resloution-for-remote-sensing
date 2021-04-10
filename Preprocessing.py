import os
import re
from skimage.transform import resize
from matplotlib import pyplot
import numpy as np
import cv2
from tqdm.notebook import tqdm, trange


def pixalate_image(image, scale_percent = 40 , downsampling_mode = cv2.INTER_AREA):

    w,h,_ = image.shape
    width = int(w * scale_percent / 100)
    height = int(h * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation = downsampling_mode )
    
    # scale back to original size
    dim = (w,h)
    low_res_image = cv2.resize(small_image, dim, interpolation =  downsampling_mode )
    return low_res_image

## Progress bar is to be added
def Data_Preprocessing(images_list ,path, Preprocessed_Data_Path , pixelation_scale = 40 ,DownSamplingMode = cv2.INTER_AREA  ):
    progress = tqdm(total= len(images_list), position=0)
    for filepath in images_list:
        filename , ext = filepath.split('/')[-1].split('.')
        image = pyplot.imread(filepath) # read the image file and save into an array
        if len(image.shape) > 2:
          # Resize the image so that every image is the same size
          HighRes = resize(image, (256, 256))
          # Add this image to the high res dataset
          # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
          LowRes = pixalate_image(HighRes , scale_percent = pixelation_scale , downsampling_mode = DownSamplingMode)
          np.save(os.path.join(Preprocessed_Data_Path, path+'_y', filename + '.npy'), HighRes)
          np.save(os.path.join(Preprocessed_Data_Path, path+'_x',filename + '.npy'), LowRes)
          os.remove(filepath)
          progress.update(1)
    print('Done ... ')                          
