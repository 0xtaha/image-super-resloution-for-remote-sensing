import keras
import numpy as np
from multiprocessing import Pool
from PIL import Image


def ImgToArr(image):
        image = image.convert("RGB")
        image = np.asarray(image, dtype=np.float32) / 255
        image = image[:,:,:3]
        return image

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_x , list_y, labels, batch_size=32, X_dim = (256,256),y_dim = (256,256) , n_channels=3,
                shuffle=True , p = Pool(10)):
        'Initialization'
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_x = list_x
        self.list_y = list_y
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.p = p
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
    
    

    def __data_generation(self, list_x_temp, list_y_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size, *self.dim, self.n_channels))

        X_temp = self.p.map(Image.open, list_x_temp)
        y_temp = self.p.map(Image.open, list_y_temp)

        X_temp = self.p.map(ImgToArr, X_temp)
        y_temp = self.p.map(ImgToArr, y_temp)
        
        X = np.array(X_temp)
        y = np.array(y_temp)
        # Generate data
        
        return X, y
        
