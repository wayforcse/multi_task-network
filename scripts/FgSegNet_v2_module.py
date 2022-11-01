#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dropout, Activation, SpatialDropout2D
from tensorflow.python.keras.layers import Conv2D, Cropping2D, UpSampling2D
from tensorflow.python.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.python.keras.layers import concatenate, add, multiply
from my_upsampling_2d import MyUpSampling2D
from instance_normalization import InstanceNormalization
import keras.backend as K



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
def acc_my(y_true, y_pred):
    void_label = -1.
    #y_true,_=tf.split(y_true, 2, axis=3, num=None)
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.reshape(y_true, [-1])
    idx = tf.where(tf.not_equal(y_true, tf.constant(void_label, dtype=tf.float32)))
    y_pred = tf.gather_nd(y_pred, idx) 
    y_true = tf.gather_nd(y_true, idx)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

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

class FgSegNet_v2_module(object):
    
    def __init__(self, lr, img_shape, scene, vgg_weights_path):
        self.lr = lr
        self.img_shape = img_shape
        self.scene = scene
        self.vgg_weights_path = vgg_weights_path
        self.method_name = 'FgSegNet_v2'
        
    def VGG16(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        a = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        b = x
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    
        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    
        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Dropout(0.5, name='dr1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Dropout(0.5, name='dr2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Dropout(0.5, name='dr3')(x)
        
        return x, a, b
    def VGG16_no_conva(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_2', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_2')(x)
        a = x
  
        return a
    def VGG16_no_convb(self, x): 
        
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1_3', data_format='channels_last')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2_3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_2')(x)
    
        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1_3')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2_3')(x)
        b = x
        # Block 3

    
        # Block 4       
        return b
    
    def M_FPM(self, x,x_2,x_4):
        
        
        d1 = Conv2D(64, (3, 3), padding='same')(x)        
        y = concatenate([x, d1], axis=-1, name='cat4')
        y = Activation('relu')(y)
        d4 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y)
        y = concatenate([x, d4], axis=-1, name='cat8')
        y = Activation('relu')(y)
        d8 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y)
        y = concatenate([x, d8], axis=-1, name='cat16')
        y = Activation('relu')(y)
        d16 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y)    

        
        
        maxpool_x1 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool1',padding='same')(x)
        maxpool_x1 = Conv2D(64, (1, 1), padding='same')(maxpool_x1)

    
        maxpool_x2 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool2',padding='same')(x_2)
        maxpool_x2 = Conv2D(64, (1, 1), padding='same')(maxpool_x2)

        maxpool_x4 = MaxPooling2D((2, 2), strides=(1, 1), name='max_pool4',padding='same')(x_4)
        maxpool_x4 = Conv2D(64, (1, 1), padding='same')(maxpool_x4)
        
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x_2)
        x_2 = Dropout(0.25, name='dr2_x2')(x_2)        
        x_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x_2)
        x_2 = Dropout(0.25, name='dr2_x2_2')(x_2)

        
        d1_x_2 = Conv2D(64, (3, 3), padding='same')(x_2)
        y_x_2 = concatenate([x, d1_x_2], axis=-1, name='cat4_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d4_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y_x_2)     
        y_x_2 = concatenate([x, d4_x_2], axis=-1, name='cat8_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d8_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y_x_2)
        y_x_2 = concatenate([x, d8_x_2], axis=-1, name='cat16_x_2')
        y_x_2 = Activation('relu')(y_x_2)
        d16_x_2 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y_x_2) 


        x_4=Conv2D(64, (3, 3),  activation='relu',padding='same')(x_4)
        x_4 = Dropout(0.25, name='dr2_x4')(x_4) 
        x_4=  Conv2D(64, (3, 3),  activation='relu',padding='same')(x_4)
        x_4 = Dropout(0.25, name='dr2_x4_2')(x_4) 

        
        d1_x_4 = Conv2D(64, (3, 3), padding='same')(x_4)     
        y_x_4 = concatenate([x, d1_x_4], axis=-1, name='cat4_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d4_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=2)(y_x_4)     
        y_x_4 = concatenate([x, d4_x_4], axis=-1, name='cat8_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d8_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=4)(y_x_4)
        y_x_4 = concatenate([x, d8_x_4], axis=-1, name='cat16_x_4')
        y_x_4 = Activation('relu')(y_x_4)
        d16_x_4 = Conv2D(64, (3, 3), padding='same', dilation_rate=8)(y_x_4)

        
        x_r = concatenate([d1, d4, d8, d16,maxpool_x1], axis=-1,name='xr')
        x_r = InstanceNormalization()(x_r)
        x_r = Activation('relu')(x_r)
        x_r = SpatialDropout2D(0.25)(x_r)
        
        x_r_2 = concatenate([d1_x_2, d4_x_2, d8_x_2 ,d16_x_2,maxpool_x2], axis=-1,name='xr_2')
        x_r_2 = InstanceNormalization()(x_r_2)
        x_r_2 = Activation('relu')(x_r_2)
        x_r_2 = SpatialDropout2D(0.25)(x_r_2)   
        
        x_r_4 = concatenate([d1_x_4, d4_x_4, d8_x_4 ,d16_x_4,maxpool_x4], axis=-1,name='xr_4')
        x_r_4 = InstanceNormalization()(x_r_4)
        x_r_4 = Activation('relu')(x_r_4)
        x_r_4 = SpatialDropout2D(0.25)(x_r_4)           
        
        return x_r, x_r_2, x_r_4    
    
    def mydecoder(self,x,a,b,x_size2,x_size4):


        
        a = GlobalAveragePooling2D()(a)
        b = Conv2D(64, (1, 1), strides=1, padding='same')(b)
        
        
        b = GlobalAveragePooling2D()(b)

        shared_conv_4 = Conv2D(64, (3,3), padding='same')            
        x = shared_conv_4(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)        
        
        x_size2 = shared_conv_4(x_size2)
        x_size2 = InstanceNormalization()(x_size2)
        x_size2 = Activation('relu')(x_size2)

        x_size4 = shared_conv_4(x_size4)
        x_size4 = InstanceNormalization()(x_size4)
        x_size4 = Activation('relu')(x_size4)   
        
        x_4 = concatenate([x, x_size2,x_size4], axis=-1, name='cat_x_4_decoder')
                
        #output
        x1 = multiply([x, b])
        x = add([x, x1])
        
        x_size22 = multiply([x_size2, b])
        x_size2 = add([x_size2, x_size22])
        
        x_size42 = multiply([x_size4, b])
        x_size4 = add([x_size4, x_size42])
        

        x = UpSampling2D(size=(2, 2))(x)
        x_size2 = UpSampling2D(size=(2, 2))(x_size2)
        x_size4 = UpSampling2D(size=(2, 2))(x_size4)
        
        
        
        shared_conv_2 = Conv2D(64, (3,3), padding='same')
        x = shared_conv_2(x)                
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
        
        x_size2 = shared_conv_2(x_size2)                
        x_size2 = InstanceNormalization()(x_size2)
        x_size2 = Activation('relu')(x_size2)
        
        
        x_size4 = shared_conv_2(x_size4)                
        x_size4 = InstanceNormalization()(x_size4)
        x_size4 = Activation('relu')(x_size4)
        
        
        x_2 = concatenate([x, x_size2,x_size4], axis=-1, name='3-size-concate_2')
        
        
        
        
        x2 = multiply([x, a])
        x = add([x, x2])   
        
        x_size42 = multiply([x_size4, a])
        x_size4 = add([x_size4, x_size42])         
        
        x_size22 = multiply([x_size2, a])
        x_size2 = add([x_size2, x_size22])          
        
        x = UpSampling2D(size=(2, 2))(x)
        x_size2 = UpSampling2D(size=(2, 2))(x_size2)
        x_size4 = UpSampling2D(size=(2, 2))(x_size4)
        
        shared_conv_1 = Conv2D(64, (3,3), padding='same')
        
        
        x_size4 = shared_conv_1(x_size4)
        x_size4 = InstanceNormalization()(x_size4)
        x_size4 = Activation('relu')(x_size4)
        
        x_size2 = shared_conv_1(x_size2)
        x_size2 = InstanceNormalization()(x_size2)
        x_size2 = Activation('relu')(x_size2)
        
        
        x = shared_conv_1(x) 
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        x = concatenate([x, x_size2,x_size4], axis=-1, name='3-size-concate')     
         
        return x,x_2,x_4
    
    

    
    def initModel(self, dataset_name):
        print('goin')
        assert dataset_name in ['CDnet', 'SBI', 'UCSD'], 'dataset_name must be either one in ["CDnet", "SBI", "UCSD"]]'
        assert len(self.img_shape)==3
        h, w, d = self.img_shape
        
        net_input = Input(shape=(h, w, d), name='net_input')
        print(net_input.shape)
        x2_input = Input(shape=(int(h/2), int(w/2), d), name='x2_input')
        x4_input = Input(shape=(int(int(h/2)/2), int(int(w/2)/2), d), name='x4_input')
 
        

        #model5 = Model(inputs=superpixel_shape, outputs=superpixel_shape, name='model5')

        
        
        vgg_output = self.VGG16(net_input)
        vgg_output2 = self.VGG16_no_convb(x2_input)
        vgg_output3 = self.VGG16_no_conva(x4_input)
        
        model = Model(inputs=net_input, outputs=vgg_output, name='model')       
        model2 = Model(inputs=x2_input,outputs=vgg_output2, name='model2')
        model3 = Model(inputs=x4_input,outputs=vgg_output3, name='model3')
        
        model.load_weights(self.vgg_weights_path, by_name=True)
        model2.load_weights(self.vgg_weights_path, by_name=True)
        model3.load_weights(self.vgg_weights_path, by_name=True)
        
        unfreeze_layers = ['block4_conv1','block4_conv2', 'block4_conv3']
        for layer in model.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable  = False
        for layer in model2.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False
        for layer in model3.layers:
            if(layer.name not in unfreeze_layers):
                layer.trainable = False                
        x,a,b = model.output
        
                #new here
        
        vgg_output2 = model2(x2_input)
        x2_b = vgg_output2

        vgg_output3 = model3(x4_input)       
        x4_a = vgg_output3
    
        
        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            x1_ups = {'streetCornerAtNight':(0,1), 'tramStation':(1,0), 'turbulence2':(1,0)}
            for key, val in x1_ups.items():
                if self.scene==key:
                    # upscale by adding number of pixels to each dim.
                    x = MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x)
                    x2_b=MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x2_b)
                    x4_a=MyUpSampling2D(size=(1,1), num_pixels=val, method_name = self.method_name)(x4_a)
                    break
                
        x,x_size2_FPM,x_size4_FPM = self.M_FPM(x,x2_b,x4_a)
        
        x,x_2,x_4 = self.mydecoder(x,a,b,x_size2_FPM,x_size4_FPM)
      
        
        x_4 = Conv2D(64, (3, 3), strides=1, padding='same')(x_4)
        x_4 = InstanceNormalization()(x_4)
        x_4 = Activation('relu')(x_4)
        x_4 = Conv2D(1, 1, padding='same', activation='sigmoid',name='output_x_4')(x_4)
        
        x_2 = Conv2D(64, (3, 3), strides=1, padding='same')(x_2)
        #x_4_to_x2=UpSampling2D(size=(2, 2))(x_4)
        #x_2 = concatenate([x_2,x_4_to_x2], axis=-1, name='3-size-concate-x2-fianl')   
        x_2 = InstanceNormalization()(x_2)
        x_2 = Activation('relu')(x_2)
        x_2 = Conv2D(1, 1, padding='same', activation='sigmoid',name='output_x_2')(x_2)
       

        #x_2_to_ori=UpSampling2D(size=(2, 2))(x_2)
        #x_4_to_ori=UpSampling2D(size=(4, 4))(x_4)         
        x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
        #x = concatenate([x, x_2_to_ori,x_4_to_ori], axis=-1, name='3-size-concate-final')   
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)
        
        

        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                tosetname='output_x_before'
            elif(self.scene=='bridgeEntry'):
                tosetname='output_x_before'
            elif(self.scene=='fluidHighway'):
                tosetname='output_x_before'
            elif(self.scene=='streetCornerAtNight'): 
                tosetname='output_x_before'
            elif(self.scene=='tramStation'):  
                tosetname='output_x_before'
            elif(self.scene=='twoPositionPTZCam'):
                tosetname='output_x_before'
            elif(self.scene=='turbulence2'):
                tosetname='output_x_before'
            elif(self.scene=='turbulence3'):
                tosetname='output_x_before'
            else:
                tosetname='output_x'
        else:
            tosetname='output_x'

        
        x = Conv2D(1,1, padding='same', activation='sigmoid',name=tosetname)(x) 
        

        verifyusepad='yes'
        # pad in case of CDnet2014
        if dataset_name=='CDnet':
            if(self.scene=='tramCrossroad_1fps'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='bridgeEntry'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,2), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='fluidHighway'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='streetCornerAtNight'): 
                x = MyUpSampling2D(size=(1,1), num_pixels=(1,0), method_name=self.method_name)(x)
                x = Cropping2D(cropping=((0, 0),(0, 1)),name='output_x')(x)
            elif(self.scene=='tramStation'):  
                x = Cropping2D(cropping=((1, 0),(0, 0)),name='output_x')(x)
            elif(self.scene=='twoPositionPTZCam'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,2), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='turbulence2'):
                x = Cropping2D(cropping=((1, 0),(0, 0)))(x)
                x = MyUpSampling2D(size=(1,1), num_pixels=(0,1), method_name=self.method_name,name='output_x')(x)
            elif(self.scene=='turbulence3'):
                x = MyUpSampling2D(size=(1,1), num_pixels=(2,0), method_name=self.method_name,name='output_x')(x)
            else:
                verifyusepad='no'


        print('verifyusepadding:'+(verifyusepad))
        vision_model = Model(inputs=[net_input,x2_input,x4_input], outputs=[x,x_2,x_4], name='vision_model')
        opt = tf.keras.optimizers.RMSprop(lr = self.lr, rho=0.9, epsilon=1e-08, decay=0.)

        # Since UCSD has no void label, we do not need to filter out
        if dataset_name == 'UCSD':
            c_loss = loss2
            c_acc = acc2
        else:
            c_loss = loss
            c_acc = acc

        losses ={'output_x':loss,
                 'output_x_2':loss,
                 'output_x_4':loss,

        }
        
        lossWeights={'output_x':1,
                     'output_x_2':1,
                     'output_x_4':1,
         
        }
        accs={'output_x':c_acc,
                     'output_x_2':c_acc,
                     'output_x_4':c_acc,
         
        }
 


        vision_model.compile(loss=losses,loss_weights=lossWeights, optimizer=opt, metrics=accs)



        return vision_model
    
