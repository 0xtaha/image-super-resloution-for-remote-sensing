import os
from skimage.transform import resize
from matplotlib import pyplot
import numpy as np
import cv2
from tqdm.notebook import tqdm, trange


def pixalate_image(image, resize_dim = (96 , 96) , downsampling_mode = cv2.INTER_AREA , same_size = True):

    w,h,_ = image.shape
    
    if (downsampling_mode == None ):
        small_image = cv2.resize(image, resize_dim )
    else:
        small_image = cv2.resize(image, resize_dim, interpolation = downsampling_mode )
    
    # scale back to original size
    if (same_size == False) :
        return small_image

    resize_dim = (w,h)
    if (downsampling_mode == None ):
        low_res_image = cv2.resize(small_image, resize_dim )
    else:
        low_res_image = cv2.resize(small_image, resize_dim, interpolation = downsampling_mode )

    return low_res_image

## Progress bar is to be added
def Data_Preprocessing(images_list ,path, Preprocessed_Data_Path , resize_dim = (96 , 96) ,DownSamplingMode = cv2.INTER_AREA  ):
    progress = tqdm(total= len(images_list), position=0)
    for i , filepath in enumerate(images_list):
        image = pyplot.imread(filepath) # read the image file and save into an array
        if len(image.shape) > 2:
          # Resize the image so that every image is the same size
          HighRes = resize(image, (256, 256))
          # Add this image to the high res dataset
          # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
          LowRes = pixalate_image(HighRes , resize_dim , downsampling_mode = DownSamplingMode)
          name = "{}".format("{0:05d}".format(i))
          np.save(os.path.join(Preprocessed_Data_Path, path+'_y', name + '.npy'), HighRes)
          np.save(os.path.join(Preprocessed_Data_Path, path+'_x',name + '.npy'), LowRes)
          os.remove(filepath)
          progress.update(1)
    print('Done ... ')                          
