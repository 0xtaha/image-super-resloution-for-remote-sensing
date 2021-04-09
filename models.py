from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.backend import clear_session
from tensorflow.keras import Sequential

def Autoencoder(img_shape = (256, 256, 3)):
    clear_session()

    Input_img = Input(shape=img_shape)  

    #encoding architecture
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)
    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)

    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
    #encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    # decoding architecture
    x7 = UpSampling2D()(encoded)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])
    x11 = UpSampling2D()(x10)
    x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])
    # x3 = UpSampling2D((2, 2))(x3)
    # x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
    # x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
    decoded = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)
    autoencoder = Model(Input_img, decoded)
    return autoencoder

def SRCNN(img_shape = (256, 256, 3)):
    model = Sequential()
    model.add(Conv2D(32, 9, activation="relu", input_shape=img_shape, padding="same"))
    model.add(Conv2D(16, 5, activation="relu", padding="same"))
    model.add(Conv2D(3, 5, activation="relu", padding="same"))
    return model