#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dropout, Activation, SpatialDropout2D, Dense, Flatten, Reshape
from tensorflow.python.keras.layers import BatchNormalization, PReLU
from tensorflow.python.keras.layers import Conv3D, Conv3DTranspose, UpSampling3D
from tensorflow.python.keras.layers import Conv2D, Cropping2D, UpSampling2D
from tensorflow.python.keras.layers import MaxPooling3D, GlobalAveragePooling3D
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import concatenate, add, multiply
from tensorflow.python.keras.regularizers import l2
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf

def d_loss(y_true, y_pred):
    dl = K.binary_crossentropy(y_true, y_pred)
    return dl
def d_acc(y_true, y_pred):
    return K.equal(y_true, K.round(y_pred))

def loss(y_true, y_pred):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    y_true = K.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def my_loss(y_true, y_pred,verify_feature):
    void_label = -1.
    y_pred = K.reshape(y_pred, [-1])
    #y_true,superpixel=tf.split(y_true, 2, axis=3, num=None)
    y_true = K.reshape(y_true, [-1])##chanfe into Tensor
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)

    verify_feature=K.reshape(verify_feature, [-1])
    verify_feature = tf.gather_nd(verify_feature, idx) 
    

    
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)+K.mean(K.binary_crossentropy(y_true, verify_feature), axis=-1)   
def DiceBCELoss(y_true, y_pred, smooth=1e-6):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    BCE =  loss(y_true, y_pred)
    intersection = K.sum(targets * inputs)    
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE

def DiceLoss(y_true, y_pred, smooth=1e-6):
    
    #flatten label and prediction tensors
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

def acc(y_true, y_pred):
    void_label = -1.
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


def loss2(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

def acc2(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

class Vnet_module(object):
    
    def __init__(self, lr, img_shape):
        self.lr = lr
        self.img_shape = img_shape
        self.method_name = 'FgSegNet_v2'
    def encoder(self, x):
        #block 1
        x = Conv3D(32, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(64, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        a = x
        x = MaxPooling3D((2, 2, 2), strides = 2)(x)

        #block 2
        x = Conv3D(64, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(128, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        b = x
        x = MaxPooling3D((2, 2, 2), strides = 2)(x)

        #block 3
        x = Conv3D(128, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(256, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        c = x
        x = MaxPooling3D((2, 2, 2), strides = 2)(x)

        #block 4
        x = Conv3D(256, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv3D(512, (3, 3, 3), padding='same', activation = None, data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)


        return x, a, b, c

    def decoder(self, x, a, b, c):
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3D(512, (2, 2, 2), strides = 1, padding = 'same', activation = None)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #deblock 3
        x_c = concatenate([x, c], axis=-1, name = 'concat1')
        x_c = Conv3D(256, (3, 3, 3), padding='same', activation = None)(x_c)
        x_c = BatchNormalization()(x_c)
        x_c = Activation('relu')(x_c)
        x_c = Conv3D(256, (3, 3, 3), padding='same', activation = None)(x_c)
        x_c = BatchNormalization()(x_c)
        x_c = Activation('relu')(x_c)
        x = x_c
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3D(256, (2, 2, 2), strides = 1, padding = 'same', activation = None)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #deblock 2
        x_b = concatenate([x, b], axis=-1, name = 'concat2')
        x_b = Conv3D(128, (3, 3, 3), padding='same', activation = None)(x_b)
        x_b = BatchNormalization()(x_b)
        x_b = Activation('relu')(x_b)
        x_b = Conv3D(128, (3, 3, 3), padding='same', activation = None)(x_b)
        x_b = BatchNormalization()(x_b)
        x_b = Activation('relu')(x_b)
        x = x_b
        x = UpSampling3D(size=(2, 2, 2))(x)
        x = Conv3D(128, (2, 2, 2), strides = 1, padding = 'same', activation = None)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        #deblock 1
        x_a = concatenate([x, a], axis=-1, name = 'concat3')
        x_a = Conv3D(64, (3, 3, 3), padding='same', activation = None)(x_a)
        x_a = BatchNormalization()(x_a)
        x_a = Activation('relu')(x_a)
        x_a = Conv3D(32, (3, 3, 3), padding='same', activation = None)(x_a)
        x_a = BatchNormalization()(x_a)
        x_a = Activation('relu')(x_a)
        # x = add([x, x_a])
        x = x_a
        x = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid', name='frame_output')(x)

        return x

    def class_head(self, x):
        # x = GlobalAveragePooling3D(name = 'GAP')(x)
        x = Flatten()(x)
        x = Dense(256,activation='relu',kernel_regularizer=l2(0.0002))(x)
        x = Dense(256,activation='relu',kernel_regularizer=l2(0.0002))(x)
        x = Dropout(0.7)(x)
        x = Dense(1, activation='sigmoid', kernel_regularizer=l2(0.0002), name='class_output')(x)
        return x

    
    

    
    def initModel(self, dataset_name):
        print('goin')
        assert dataset_name in ['Luna16'], 'dataset_name must be either one in ["Luna16"]'
        assert len(self.img_shape)==4
        h, w, d, c= self.img_shape
        
        net_input = Input(shape=(h, w, d, c), name='net_input')
        encoder_output= self.encoder(net_input)
        model = Model(inputs=net_input, outputs=encoder_output, name='model')
        x, a, b, c = model.output
        # print(x.shape)
        # print(a.shape)
        # print(b.shape)
        # print(c.shape)
        # print(d.shape)
        x = self.decoder(x, a, b, c)
        class_output = self.class_head(x)
        print(x.shape)
      
        
        vision_model = Model(inputs=net_input, outputs=[x,class_output], name='vision_model')
        opt = keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        c_loss = d_loss
        f_loss = DiceBCELoss
        c_acc = d_acc
        f_acc = acc

        losses ={'frame_output':f_loss,
                'class_output':c_loss}
        lossWeights={'frame_output':1,
                'class_output':1}
        accs={'frame_output':f_acc,
                'class_output':c_acc}
 


        vision_model.compile(loss=losses,loss_weights=lossWeights, optimizer=opt, metrics=accs)


        return vision_model
    
