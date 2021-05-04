import os
from skimage.transform import resize
from matplotlib import pyplot
import numpy as np
import cv2
from tqdm.notebook import tqdm
from multiprocessing import Pool
from itertools import repeat


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

def image_split(path_to_img, savepath ,split_width, split_height , overlap_x=0, overlap_y=0 , frmt='png' , counter = 0):
    """
    overlap --> 0 : 0.75
    """
    img = cv2.imread(path_to_img, cv2.IMREAD_COLOR)
    img_h, img_w, _ = img.shape
    
    X_points = start_points(img_w, split_width, overlap_x)
    Y_points = start_points(img_h, split_height, overlap_y)

    count = counter

    open(path_to_img, 'w').close() #overwrite and make the file blank instead - ref: https://stackoverflow.com/a/4914288/3553367
    os.remove(path_to_img)

    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            name = "{}.{}".format("{0:08d}".format(count),frmt)
            pyplot.imsave(os.path.join(savepath, name), split, format = frmt)
            count += 1
    return count


def pixalate_image(image, pixelation_scale = 0.5 , same_size = True):

    w,h,_ = image.shape
    resize_dim = (int(w*pixelation_scale) , int(h*pixelation_scale))
    small_image = cv2.resize(image, resize_dim, interpolation = cv2.INTER_AREA )
    
    # scale back to original size
    if (same_size == False) :
        return small_image

    resize_dim = (w,h)

    low_res_image = cv2.resize(small_image, resize_dim )
    return low_res_image

def image_preprocess(filepath , Preprocessed_Data_Path , path , pixelation_scale = 0.5):
    image = cv2.imread(filepath ,cv2.IMREAD_COLOR) # read the image file and save into an array
    if len(image.shape) > 2:
        # Resize the image so that every image is the same size
        HighRes = resize(image, (256, 256))
        # Add this image to the high res dataset
        # Rescale it 0.5x and 2x so that it is a low res image but still has 256x256 resolution
        LowRes = pixalate_image(HighRes , pixelation_scale = pixelation_scale)
        
        name = (os.path.split(filepath)[1]).split('.')[0]
        
        pyplot.imsave(os.path.join(Preprocessed_Data_Path, path+'_y', name + '.png'), HighRes)
        pyplot.imsave(os.path.join(Preprocessed_Data_Path, path+'_x',name + '.png'), LowRes)

## Progress bar is to be added
def Data_Preprocessing(images_list , Preprocessed_Data_Path , path , pixelation_scale = 0.5):
    list_len = len(images_list)
    p = Pool(10)

    pr = repeat(Preprocessed_Data_Path , list_len)
    pa = repeat(path, list_len)
    pi = repeat(pixelation_scale, list_len)

    p.starmap(image_preprocess, zip(images_list, pr , pa  , pi))

    # while(list_len > begin):
    #     current_processed_images = images_list[begin : begin+number_of_threads]
    #     begin +=number_of_threads
    #     for image in current_processed_images:
    #         image_preprocess(image , Preprocessed_Data_Path , path , pixelation_scale )
    #     # p.starmap(image_preprocess, zip(current_processed_images , pr , pa  , pi))
    #     progress.update(number_of_threads)
    print('Done ... ')