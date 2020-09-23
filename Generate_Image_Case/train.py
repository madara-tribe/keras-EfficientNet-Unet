from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow==1.13.1
!pip install keras==2.2.4

!unzip layers.zip

import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import keras
import os
import cv2
from keras.optimizers import Adam, RMSprop, SGD
from efficientunet import _get_efficient_unet
from efficientnet import get_efficientnet_b5_encoder
from sklearn.metrics import mean_absolute_error
from skimage.measure import compare_ssim, compare_psnr
import keras.backend as K

def measurement(func, **kwargs):
    val = func(kwargs["img1"], kwargs["img2"])
    print(val)
    return val

def eval(model, X, Y):
    y_pred = model.predict(X)
    print(y_pred.shape)
    
    num_data = len(y_pred)
    maes= 0
    psnrs = 0
    mae = 0
    for idx, (pr, gt) in enumerate(zip(y_pred, Y)):
        print(pr.shape, gt.shape)
        print('prediction')
        plt.imshow(pr),plt.show()
        print('GT')
        plt.imshow(gt),plt.show()
        pr1d = pr.flatten().reshape(1, H*H*3)
        gt1d=gt.flatten().reshape(1, H*H*3)
        print(idx, mean_absolute_error(pr1d, gt1d))
        mae += np.sum(np.absolute((gt - pr)))
        maes += mean_absolute_error(pr1d, gt1d)
        #psnrs += measurement(compare_psnr, img1=pr1d, img2=gt1d)
        #ssims += measurement(compare_ssim, img1=im, img2=pr)

    print('total mae', mae/num_data)
    #print('total psnr', psnrs/num_data)
    print('total maes', maes/num_data)


def load_model(H):
    input_shape = (H, H, 3)
    out_channels = 3

    encoder = get_efficientnet_b5_encoder(input_shape, pretrained=False)
    model = _get_efficient_unet(encoder, out_channels=out_channels, 
                          concat_input=True, fpa=None, hypercolumn=None)
 sgd = keras.optimizers.SGD(lr=1E-2, decay=5**(-4), momentum=0.9, nesterov=True)
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.001))
    model.summary()
    return model







EPOCH=100
H=128
print(H)

print('loading train data')
X_train = np.load('/content/drive/My Drive/masked.npy')
X_train = np.array([cv2.resize(img, (128, 128)) for img in X_train])

plt.imshow(X_train[1]),plt.show()
Xs, valX = X_train[:10], X_train[10:15]
print(Xs.shape, valX.shape, Xs.max(), Xs.min())

print('loading validation data')
Y_train = np.load('/content/drive/My Drive/ori.npy')
Y_train = np.array([cv2.resize(img, (128, 128)) for img in Y_train])


Ys, valY = Y_train[:10], Y_train[10:15]
print(Ys.shape, valY.shape, Ys.max(), Ys.min())
plt.imshow(Y_train[1]),plt.show()



print('load model')
checkpoint_path = "drive/My Drive/train_ck/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, period=1)

model = load_model(H)


"""train"""
hist1 = model.fit(x=Ys, y=Xs, batch_size=2, epochs=100,  
            validation_data=(valY, valX))
for key in ["loss", "val_loss"]:
    plt.plot(hist1.history[key],label=key)
plt.legend()



print('evalation')
eval(model, valX, valY)

