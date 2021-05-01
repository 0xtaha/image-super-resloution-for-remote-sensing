import os
from skimage.transform import resize
from matplotlib import pyplot
import numpy as np
import cv2
from tqdm.notebook import tqdm, trange

def start_points(size, split_size, overlap=0):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

def image_split(path_to_img, savepath ,split_width, split_height , overlap_x=0, overlap_y=0 , format='png'):
    """
    overlap --> 0 : 0.75
    """
    img = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    
    X_points = start_points(img_w, split_width, overlap_x)
    Y_points = start_points(img_h, split_height, overlap_y)

    count = 0
    name = 'splitted'
    frmt = format

    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            print('{}/{}_{}.{}'.format(savepath,name, count, frmt))
            cv2.imwrite(savepath, '{}/{}_{}.{}'.format(savepath,name, count, frmt), split)
            count += 1

def pixalate_image(image, resize_dim = (256 , 256) , downsampling_mode = cv2.INTER_AREA , same_size = True):

    w,h,_ = image.shape
    
    if (downsampling_mode == None ):
        small_image = cv2.resize(image, resize_dim )
    else:
        small_image = cv2.resize(image, resize_dim, interpolation = downsampling_mode )
    
    # scale back to original size
    if (same_size == False) :
        return small_image

    resize_dim = (w,h)
    low_res_image = cv2.resize(small_image, resize_dim )

    return low_res_image

## Progress bar is to be added
def Data_Preprocessing(images_list ,path, Preprocessed_Data_Path , resize_dim = (100 , 100) , DownSamplingMode = cv2.INTER_AREA):
    progress = tqdm(total= len(images_list), position=0)
    for i , filepath in enumerate(images_list):
        image = cv2.imread(filepath ,cv2.IMREAD_COLOR) # read the image file and save into an array
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
