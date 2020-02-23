from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import keras
import os
import cv2
from keras.optimizers import Adam, RMSprop, SGD
from EfficientUnet.efficientunet import _get_efficient_unet
from EfficientUnet.efficientnet import get_efficientnet_b5_encoder

from sklearn.metrics import mean_absolute_error
from skimage.measure import compare_ssim, compare_psnr
def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    print(val)
    return val




def evaluate(model, test_X, test_met, test_sat, test_Y):
  print('evalation')

  fp = model.predict([test_X, test_met, test_sat])
  print(fp.shape)

  y_pred = fp.reshape(24, 128, 128)
  test = test_Y.reshape(24, 128, 128)

  num_data = len(y_pred)
  maes= 0
  psnrs = 0
  ssims = 0
  for idx, (pr, im) in enumerate(zip(y_pred, test)):
    plt.imshow(pr, 'gray'),plt.show()
    plt.imshow(im, 'gray'),plt.show()
    print(idx, mean_absolute_error(pr, im))
    maes += mean_absolute_error(pr, im)
    psnrs += measurement(compare_psnr, img1=im, img2=pr)
    ssims += measurement(compare_ssim, img1=im, img2=pr)

  print('total mae', maes/num_data)
  print('total psnr', psnrs/num_data)
  print('total ssim', ssims/num_data)



def train():
  # load train daata
  X = np.load('/content/drive/My Drive/train_unet/train_X.npy')
  X = X.reshape(10, 128, 128, 1)
  Xs = X/255
  print(Xs.max(), Xs.min(), Xs.shape)

  y = np.load('/content/drive/My Drive/train_unet/target_X.npy')
  Y = y.reshape(10, 128, 128, 24)
  Ys = Y/255
  print(Ys.max(), Ys.min(), Ys.shape)


  sat = np.load('/content/drive/My Drive/train_unet/sat_X.npy')
  sat = sat.reshape(10, 128, 128, 8)
  sats = sat/255
  print(sats.shape)

  wind = np.load('/content/drive/My Drive/train_unet/wind_X.npy')
  wind = wind.reshape(10, 64, 64, 8)
  winds = wind/255
  print(winds.shape)



  # load validation data
  v_X = np.load('/content/drive/My Drive/valid_unet/valid_X.npy')
  valid_X = v_X.reshape(1, 128, 128, 1)
  valid_X = valid_X/255

  v_y = np.load('/content/drive/My Drive/valid_unet/valid_target_X.npy')
  valid_Y = v_y.reshape(1, 128, 128, 24)
  valid_Y = valid_Y/255

  v_sat = np.load('/content/drive/My Drive/valid_unet/valid_sat.npy')
  valid_sat = v_sat.reshape(1, 128, 128, 8)
  valid_sat = valid_sat/255

  v_wind = np.load('/content/drive/My Drive/valid_unet/valid_wind.npy')
  valid_wind = v_wind.reshape(1, 64, 64, 8)
  valid_wind = valid_wind/255

  print(valid_X.shape, valid_Y.shape, valid_sat.shape, valid_wind.shape)



  # load model
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  input_shape = (128, 128, 1)
  met_shape = (64, 64, 8)
  sat_shape = (128, 128, 8)
  out_channels = 24

  encoder = get_efficientnet_b5_encoder(input_shape, pretrained=False)
  model = _get_efficient_unet(encoder, met_shape, sat_shape, out_channels=out_channels, 
                              concat_input=True, fpa=None, hypercolumn=True)
  model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
  model.summary()

  # train
  model.fit(x=[Xs, winds, sats], y=Ys, batch_size=2, epochs=100000,  
            validation_data=([valid_X, valid_wind, valid_sat], valid_Y))
    
  return model

  
def predict(model):
  # load test data
  test_wind = np.load('/content/drive/My Drive/test_unet/test2/wind2.npy')
  test_wind = test_wind.reshape(1, 64, 64, 8)
  test_winds = test_wind/255

  test_sat = np.load('/content/drive/My Drive/test_unet/test2/test_sat2.npy')
  test_sat = test_sat.reshape(1, 128, 128, 8)
  #test_sat[test_sat<63]=0
  #test_sat[test_sat>220]=255
  test_sats = test_sat/255
  test_sats.shape
  print(test_winds.shape, test_sats.shape, test_sats.min(), test_sats.max())


  test_Y = np.load('/content/drive/My Drive/test_unet/test2/test_Y2.npy')
  tX = '2018-08-08-23-00.fv.png'
  tX = cv2.imread(tX, 0)
  test_X = cv2.resize(tX, (128, 128))
  test_X = test_X.reshape(1, 128, 128, 1)
  #test_X[test_X<63]=0
  #test_X[test_X>220]=255
  test_X = test_X/255
  print(test_X.shape, test_Y.shape, test_X.max(), test_X.min())

  evaluate(model, test_X, test_winds, test_sats, test_Y)


if __name__ == '__main__':
  model = train()
  predict(model)