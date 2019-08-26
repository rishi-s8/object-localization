import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import cv2
import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

path_train = './Lol/'
#path_test = './test/'

train_data = pd.read_csv("training.csv", dtype={"name": str, "x1": float, "x2": float, "y1": float, "y2": float})
#test_data = pd.read_csv("test.csv", dtype={"name": str})

def Encoder(input_img):
	Econv1_1 = Conv2D(16, (3, 3), activation='sigmoid', padding='same', name='block1_conv1')(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='sigmoid', padding='same', name='block1_conv2')(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool1')(Econv1_2)

	Econv2_1 = Conv2D(32, (3, 3), activation='sigmoid', padding='same', name='block2_conv1')(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(32, (3, 3), activation='sigmoid', padding='same', name='block2_conv2')(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool1')(Econv2_2)

	Econv3_1 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='block3_conv1')(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='block3_conv2')(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_pool1')(Econv3_2)

	Econv4_1 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='block4_conv1')(pool3)
	Econv4_1 = BatchNormalization()(Econv4_1)
	Econv4_2 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name='block4_conv2')(Econv4_1)
	Econv4_2 = BatchNormalization()(Econv4_2)
	pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_pool1')(Econv4_2)

	encoded = Model(inputs=input_img, outputs=pool4)
	return encoded

def Regressor(input_img):
	reg_conv1_1 = Conv2D(16, (3, 3), activation = 'sigmoid', padding= 'same', name = 'block1_conv1')(input_img)
	reg_conv1_1 = BatchNormalization()(reg_conv1_1)
	reg_conv1_2 = Conv2D(16, (3, 3), activation = 'sigmoid', padding = 'same', name= 'block1_conv2')(reg_conv1_1)
	reg_conv1_2 = BatchNormalization()(reg_conv1_2)
	reg_pool1 = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding='same', name= "block1_pool1")(reg_conv1_2)

	reg_conv2_1 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name = "block2_conv1")(reg_pool1)
	reg_conv2_1 = BatchNormalization()(reg_conv2_1)
	reg_conv2_2 = Conv2D(64, (3, 3), activation='sigmoid', padding='same', name = "block2_conv2")(reg_conv2_1)
	reg_conv2_2 = BatchNormalization()(reg_conv2_2)
	reg_pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(reg_conv2_2)

	reg_conv3_1 = Conv2D(128, (3, 3), activation='sigmoid', padding='same', name = "block3_conv1")(reg_pool2)
	reg_conv3_1 = BatchNormalization()(reg_conv3_1)
	reg_conv3_2 = Conv2D(128, (3, 3), activation='sigmoid', padding='same', name = "block3_conv2")(reg_conv3_1)
	reg_conv3_2 = BatchNormalization()(reg_conv3_2)
	reg_pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(reg_conv3_2)

	reg_flat = Flatten()(input_img)
	fc1 = Dense(256, activation='sigmoid')(reg_flat)
	fc2 = Dense(64, activation='sigmoid')(fc1)
	fc3 = Dense(16, activation='sigmoid')(fc2)
	fc4 = Dense(4, activation='sigmoid')(fc3)
	regress = Model(inputs = input_img, outputs =  fc4)
	return regress


x_shape = 480
y_shape = 640
channels = 3

input_img = Input(shape = (x_shape,y_shape,channels))

#Encoder
encoded = Encoder(input_img)
encoded.summary()

Regressor_input = Input(shape = (int(x_shape/(2**4)),int(y_shape/(2**4)),64))
regressor = Regressor(Regressor_input)

output_image = regressor(encoded.outputs)

model = Model(inputs = input_img, outputs = output_image)

model.compile(loss='mean_squared_error', optimizer = 'adadelta', metrics=['mse'])

model.summary()

model.load_weights('./newRegressor/finalEpoch.h5')

encoded.trainable = True
regressor.trainable = True

filepath='./newRegressor/2model-full.h5'
filepath2='./newRegressor/2finalEpoch.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


epochs = 10
loss=[]
val_loss=[]
for e in range(epochs):
    for y in range(0,14000, 1000):
        x_train = []
        y_train = []

        for elem in (list(train_data.itertuples(index=False)))[y:y+1000]:
            path = path_train + str(elem[0])
            print(path)
            img = cv2.imread(path)
            x_train.append(img)
            y_train.append([elem[1]/y_shape, elem[2]/y_shape, elem[3]/x_shape, elem[4]/x_shape])

        x_train = np.asarray(x_train, dtype='float32')
        x_train /= 255
        y_train = np.asarray(y_train, dtype='float32')

        history=model.fit(x=x_train, y=y_train, batch_size=10, epochs=1, validation_split=0.05, shuffle=True, verbose=1, callbacks=callbacks_list)
        loss.append(history.history['loss'][0])
        val_loss.append(history.history['val_loss'][0])
    plt.clf()
    plt.set_yscale = ('log')
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('./newRegressor/2polots/' + str(e) +'.png')

    model.save_weights(filepath2)
