import keras
import numpy as np
from multiprocessing import Pool
from itertools import repeat
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_x , list_y, labels, batch_size=32, X_dim=(256,256), Y_dim=(256,256) , n_channels=3,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_x = list_x
        self.list_y = list_y
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_x) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_x_temp = [self.list_x[k] for k in indexes]
        list_y_temp = [self.list_y[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_x_temp , list_y_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_x))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def process_image(dim ,img):
        image = cv2.resize(dim,img)
        return image

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size, *self.dim, self.n_channels))

        p = Pool(self.batch_size)

        temp_X = p.map(np.load, list_x_temp)
        temp_y = p.map(np.load, list_y_temp)

        temp_X_imgs = p.map(self.process_image , zip( temp_X , (repeat(self.X_dim))))
        temp_y_imgs = p.map(self.process_image , zip( temp_y , (repeat(self.Y_dim))))

        X = np.array(temp_X_imgs)
        y = np.array(temp_y_imgs)

        
        # Generate data
        # for i in range (len(list_x_temp)):

        #     # Store sample
        #     X[i] = np.load(list_x_temp[i])

        #     # Store class
        #     y[i] = np.load(list_y_temp[i])
        return X, y
        
