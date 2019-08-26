import os
import pandas as pd
import numpy as np
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="1"

path_train = './train/'

train_data = pd.read_csv("training.csv", dtype={"name": str, "x1": int, "x2": int, "y1": int, "y2": int})

def Encoder(input_img):
	Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2')(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block1_pool1')(Econv1_2)

	Econv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block2_pool1')(Econv2_2)

	Econv3_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block3_pool1')(Econv3_2)

	Econv4_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3)
	Econv4_1 = BatchNormalization()(Econv4_1)
	Econv4_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(Econv4_1)
	Econv4_2 = BatchNormalization()(Econv4_2)
	pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='block4_pool1')(Econv4_2)

	encoded = Model(inputs=input_img, outputs=pool4)
	return encoded

def Decoder(input_img):
	up1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block1_up1')(input_img)
	up1 = BatchNormalization()(up1)
	Dconv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(up1)
	Dconv1_1 = BatchNormalization()(Dconv1_1)
	Dconv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(Dconv1_1)
	Dconv1_2 = BatchNormalization()(Dconv1_2)

	up2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block2_up1')(Dconv1_2)
	up2 = BatchNormalization()(up2)
	Dconv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(up2)
	Dconv2_1 = BatchNormalization()(Dconv2_1)
	Dconv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(Dconv2_1)
	Dconv2_2 = BatchNormalization()(Dconv2_2)

	up3 = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block3_up1')(Dconv2_2)
	up3 = BatchNormalization()(up3)
	Dconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv1')(up3)
	Dconv3_1 = BatchNormalization()(Dconv3_1)
	Dconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv2')(Dconv3_1)
	Dconv3_2 = BatchNormalization()(Dconv3_2)

	up4 = Conv2DTranspose(8, (3, 3), strides=(2, 2), activation='relu', padding='same', name='block4_up1')(Dconv3_2)
	up4 = BatchNormalization()(up4)
	Dconv4_1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='block4_conv1')(up4)
	Dconv4_1 = BatchNormalization()(Dconv4_1)
	Dconv4_2 = Conv2D(8, (3, 3), activation='relu', padding='same', name='block4_conv2')(Dconv4_1)
	Dconv4_2 = BatchNormalization()(Dconv4_2)

	final = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='Output')(Dconv4_2)

	decoded = Model(inputs=input_img, outputs=final)
	return decoded

x_shape = 480
y_shape = 640
channels = 3

input_img = Input(shape = (x_shape,y_shape,channels))

#Encoder
encoded = Encoder(input_img)	#return encoded representation with intermediate layer Pool3(encoded), Econv1_3, Econv2_3,Econv3_3

#Decoder
HG_ = Input(shape = (x_shape/(2**4),x_shape/(2**4),64))
decoded = Decoder(HG_)

#Combined
output_img = decoded(encoded.outputs)
model= Model(inputs = input_img, outputs = output_img )
model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mse'])
model.summary()
model.load_weights('2DAutoEncoder-tuned.h5')
filepath='./2DAutoEncoder-tuned2.h5'
filepath2='./2DAutoEncoder-finalEpoch-tuned2.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

for y in range(0,13000, 1000):
	x_train = []
	#x_test = []
	y_train = []

	for elem in (list(train_data.itertuples(index=False)))[y:y+1000]:
		path = path_train + str(elem[0])
		print(path)
		img = cv2.imread(path)
		x_train.append(img)
		y_train.append([elem[1], elem[2], elem[3], elem[4]])
	#
	# for elem in test_data.itertuples(index=False):
	# 	path = path_test + str(elem[0])
	# 	img = cv2.imread(path)
	# 	x_test.append(img)

	x_train = np.asarray(x_train, dtype='float32')
	x_train /= 255
	#x_test = np.asarray(x_test, dtype='float32')
	y_train = np.asarray(y_train, dtype='float32')

	epochs = 5

	history=model.fit(x=x_train, y=x_train, batch_size=10, epochs=epochs, validation_split=0.02, shuffle=True, verbose=1, callbacks=callbacks_list)
	plt.clf()
	plt.set_yscale = ('log')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('./plots2/' + str(y) +'.png')

model.save_weights(filepath2)
model.save_weights('encoder.h5')
