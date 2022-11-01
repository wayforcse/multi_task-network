#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 2018

@author: longang
"""

#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import numpy as np
import pandas as pd
import tensorflow as tf
import random as rn
import os,sys
import PIL.Image
import pandas
from keras.utils.vis_utils import plot_model

print('GPU',tf.test.is_gpu_available())
# set current working directory
cur_dir = os.getcwd()
os.chdir(cur_dir)
sys.path.append(cur_dir)

# =============================================================================
#  For reprodocable results, from keras.io
# =============================================================================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
session_conf = tf.compat.v1.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
K.clear_session()
tf.compat.v1.set_random_seed(1234)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras, glob
from keras.preprocessing import image as kImage
from sklearn.utils import compute_class_weight
from Vnet_module import Vnet_module
from keras.utils.data_utils import get_file
import gc

# alert the user
if keras.__version__!= '2.0.6' or tf.__version__!='1.1.0' or sys.version_info[0]<3:
    print('We implemented using [keras v2.0.6, tensorflow-gpu v1.1.0, python v3.6.3], other versions than these may cause errors somehow!\n')

# Few training frames, it may fit in memory
folder_num = 10
annotation = r'E:\NCKU\way\FgSegNet_v2-master\annotations.csv'
class_annotation = r'E:\NCKU\way\annotationdetclsconvfnl_v3.csv'
local_annotation = r'E:\NCKU\way\annotations.csv'

crop_size = 80

def crop_nodule(npy, x, y, z, size):
    crop = np.zeros((80,80,80))
    if (z-size//2 >= 0 and z+size//2 < npy.shape[0]) and (y-size//2 >= 0 and y+size//2 < npy.shape[1]) and (x-size//2 >= 0 and x+size//2 < npy.shape[2]):
        crop = npy[z-size//2:z+size//2, y-size//2:y+size//2, x-size//2:x+size//2]
    else:
        if z-size//2 < 0:
            z = size//2
        elif z+size//2 >= npy.shape[0]:
            z = npy.shape[0]-size//2-1
        if y-size//2 < 0:
            y = size//2
        elif y+size//2 >= npy.shape[1]:
            y = npy.shape[1]-size//2-1
        if x-size//2 < 0:
            x = size//2
        elif x+size//2 >= npy.shape[2]:
            x = npy.shape[2]-size//2-1
        crop = npy[z-size//2:z+size//2, y-size//2:y+size//2, x-size//2:x+size//2]
    crop = np.expand_dims(crop, axis=-1)

    return crop

def getData(ct_path, lung_mask_path, nodule_mask_path, dicom_path, folder):
    print('Data preparing...')

    ann = pd.read_csv(annotation, usecols=[0,1,2,3,5])
    ann = ann.values.tolist()

    X = []
    Y = []
    C = []

    X_v = []
    Y_v = []
    C_v = []
    for i in range(folder_num):
        # if i == folder:
        #     continue
        ct_folder_path = os.path.join(ct_path, 'subset'+str(i))
        lung_mask_folder_path = os.path.join(lung_mask_path, 'subset'+str(i))
        nodule_mask_folder_path = os.path.join(nodule_mask_path, 'subset'+str(i))
        dicom_folder_path = os.path.join(dicom_path, 'subset'+str(i))

        ct_list = os.listdir(ct_folder_path)
        # lung_mask_list = os.listdir(lung_mask_folder_path)
        # nodule_mask_list = os.listdir(nodule_mask_folder_path)

        for name in ct_list:
            ct_name = os.path.join(ct_folder_path, name)
            lung_mask_name = os.path.join(lung_mask_folder_path, name)
            nodule_mask_name = os.path.join(nodule_mask_folder_path, name)
            dicom_name = os.path.join(dicom_folder_path, name.split('.npy')[0]+'.mhd')

            idx = []
            for j in range(len(ann)): 
                if ann[j][0]==name.split('.npy')[0]:
                    idx.append(j)

            ct = np.load(ct_name)
            lung_mask = np.load(lung_mask_name)
            nodule_mask = np.load(nodule_mask_name)
            lung_mask[lung_mask!=0] = 1

            ct_lung = ct*lung_mask
            ct_nodule = ct*nodule_mask
            # nodule_mask[nodule_mask!=0] = 255

            for d in idx:
                x = ann[d][1]
                y = ann[d][2]
                z = ann[d][3]

                if i == folder:
                    C_v.append(ann[d][4])
                    X_v.append(crop_nodule(ct_lung, x, y, z, crop_size))
                    Y_v.append(crop_nodule(nodule_mask, x, y, z, crop_size)) 
                else:
                    C.append(ann[d][4])
                    X.append(crop_nodule(ct_lung, x, y, z, crop_size))
                    Y.append(crop_nodule(nodule_mask, x, y, z, crop_size))
    X = np.asarray(X).astype(np.float32)
    Y = np.asarray(Y).astype(np.float32)
    C = np.asarray(C).astype(np.float32)
    X_v = np.asarray(X_v).astype(np.float32)
    Y_v = np.asarray(Y_v).astype(np.float32)
    C_v = np.asarray(C_v).astype(np.float32)

    print('Data done')
                
    return [X, Y, C, X_v, Y_v, C_v]


### training function    
def train(data, mdl_path):
    ### hyper-params
    print(data[0].shape)
    lr = 1e-4
    val_split = 0.2
    max_epoch = 100
    batch_size = 1
    ###
    img_shape = (crop_size, crop_size, crop_size, 1)
    print(data[0].dtype)
    print(data[1].dtype)

    X_train = data[0]
    Y_train = data[1]
    C_train = data[2]
    X_test = data[3]
    Y_test = data[4]
    C_test = data[5]
    
    model = Vnet_module(lr, img_shape)
    model = model.initModel('Luna16')
    print(model.summary())
    #plot_model(model, to_file='model_plot.png', show_shapes=True)  
    #model.summary()

    # make sure that training input shape equals to model output
    # input_shape = (img_shape[0], img_shape[1])
    # output_shape = (model.output[0]._keras_shape[1], model.output[0]._keras_shape[2])
    # assert input_shape==output_shape, 'Given input shape:' + str(input_shape) + ', but your model outputs shape:' + str(output_shape)

    early = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=3*1e-5, patience=20, verbose=0, mode='auto')
    redu = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto')
    model.fit(X_train, {'frame_output':Y_train},
                validation_data=(X_test, {'frame_output':Y_test}),
                epochs=max_epoch, batch_size=batch_size, 
                callbacks=[redu, early], verbose=1, shuffle = True)

    # model.fit(X_train, {'frame_output':Y_train, 'class_output':C_train},
    #             validation_data=(X_test, {'frame_output':Y_test, 'class_output':C_test}),
    #             epochs=max_epoch, batch_size=batch_size, 
    #             callbacks=[redu, early], verbose=1, shuffle = True)
    # model.fit(X_train, {'frame_output':FY_train, 'class_output':DY_train, 'GAP': DY_train}, validation_data=(X_test, {'frame_output':FY_test, 'class_output':DY_test, 'GAP': DY_test}), epochs=max_epoch, batch_size=batch_size, 
    #       verbose=1, class_weight={'frame_output':data[3]})

    model.save(mdl_path)
    K.clear_session()
    del model, data, early, redu


# =============================================================================
# Main func
# =============================================================================
#fail at nightVideos / streetCornerAtNight

# dataset = {

#             'baseline':['highway', 'pedestrians', 'office', 'PETS2006'],
#             'cameraJitter':['badminton', 'traffic', 'boulevard', 'sidewalk'],
#             'badWeather':['skating', 'blizzard', 'snowFall', 'wetSnow'],
#             'dynamicBackground':['boats', 'canoe', 'fall', 'fountain01', 'fountain02', 'overpass'],
#             'intermittentObjectMotion':['abandonedBox', 'parking', 'sofa', 'streetLight', 'tramstop', 'winterDriveway'],
#             'lowFramerate':['port_0_17fps', 'tramCrossroad_1fps', 'tunnelExit_0_35fps', 'turnpike_0_5fps'],
#             'nightVideos':['bridgeEntry', 'busyBoulvard', 'fluidHighway', 'streetCornerAtNight', 'tramStation', 'winterStreet'],
#             'PTZ':['continuousPan', 'intermittentPan', 'twoPositionPTZCam', 'zoomInZoomOut'],
#             'shadow':['backdoor', 'bungalows', 'busStation', 'copyMachine', 'cubicle', 'peopleInShade'],
#             'thermal':['corridor', 'diningRoom', 'lakeSide', 'library', 'park'],
#             'turbulence':['turbulence0', 'turbulence1', 'turbulence2', 'turbulence3']
# }





# main_dir = os.path.join('..', 'FgSegNet_v2')
# # vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# # if not os.path.exists(vgg_weights_path):
# #     WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
# #     vgg_weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
# #                                 WEIGHTS_PATH_NO_TOP, cache_subdir='models',
# #                                 file_hash='6d6bbae143d832006294945121d1f1fc')


# # =============================================================================
# num_frames = 25 # either 2 or 10 or 25 or 200 training frames
# # =============================================================================

# # assert num_frames in [2,10,25,200], 'num_frames is incorrect.'
# main_mdl_dir = os.path.join(main_dir, 'CDnet', 'models' + str(num_frames))
# for category, scene_list in dataset.items():
#     mdl_dir = os.path.join(main_mdl_dir, category)
#     if not os.path.exists(mdl_dir):
#         os.makedirs(mdl_dir)

#     for scene in scene_list:
#         print ('Training ->>> ' + category + ' / ' + scene)
        
#         train_dir = os.path.join('..', 'training_sets', 'CDnet2014_train', category, scene + str(num_frames))
#         dataset_dir = os.path.join('..', 'datasets', 'CDnet2014_dataset', category, scene)
#         # superpixel_dir = os.path.join('..', 'datasets', 'superpixel_label', category, scene)
#         print(train_dir)
#         print(dataset_dir)
#         # print(superpixel_dir)
#         data = getData(train_dir, dataset_dir)

#         mdl_path = os.path.join(mdl_dir, 'mdl_' + scene + '.h5')
#         train(data, scene, mdl_path, vgg_weights_path)
#         del data
        
#     gc.collect()
if __name__ == '__main__':
    ct_path = r'E:\NCKU\Luna\npy\luna16_ct_spacing_npy'
    lung_mask_path = r'E:\NCKU\Luna\npy\LiDC_lung_mask_spacing_npy'
    nodule_mask_path = r'E:\NCKU\Luna\npy\LiDC_c3or_mask_spacing_npy'

    dicom_path = r'E:\NCKU\way\vnet.pytorch-master\luna16\luna16_ct'

    main_dir = os.path.join('..', 'FgSegNet_v2')
    main_mdl_dir = os.path.join(main_dir, 'Luna16', 'models')

    for i in range(folder_num):
        mdl_dir = os.path.join(main_mdl_dir, 'folder' + str(i))
        if not os.path.exists(mdl_dir):
            os.makedirs(mdl_dir)

        data = getData(ct_path, lung_mask_path, nodule_mask_path, dicom_path, i)

        mdl_path = os.path.join(mdl_dir, 'mdl_' + 'folder' + str(i) + '.h5')
        train(data, mdl_path)
        del data
        
    gc.collect()
