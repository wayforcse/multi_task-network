
"""
Created on Mon Jun 27 2018

@author: longang
"""

# coding: utf-8
#get_ipython().magic(u'load_ext autotime')
import numpy as np
import os, glob, sys
from tensorflow.python.keras.preprocessing import image as kImage
#from skimage.transform import pyramid_gaussian
from tensorflow.python.keras.models import load_model
from scipy.misc import imsave#, imresize
import gc
import pandas as pd


# Optimize to avoid memory exploding. 
# For each video sequence, we pick only 1000frames where res > 400
# You may modify according to your memory/cpu spec.
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

def checkFrame(l):
    num_frames = len(X_list) # 7000
    max_cube =  # max frames to slice
    if(img[1]>=400 and len(X_list)>max_frames):
        print ('\t- Total Frames:' + str(num_frames))
        num_chunks = num_frames/max_frames
        num_chunks = int(np.ceil(num_chunks)) # 2.5 => 3 chunks
        start = 0
        end = max_frames
        m = [0]* num_chunks
        for i in range(num_chunks): # 5
            m[i] = range(start, end) # m[0,1500], m[1500, 3000], m[3000, 4500]
            start = end # 1500, 3000, 4500 
            if (num_frames - start > max_frames): # 1500, 500, 0
                end = start + max_frames # 3000
            else:
                end = start + (num_frames- start) # 2000 + 500, 2500+0
        print ('\t- Slice to:' + str(m))
        del img, X_list
        return [True, m]
    del img, X_list
    return [False, None]
    
# Load some frames (e.g. 1000) for segmentation
def generateData(scene_input_path, X_list, scene):
    # read images
    X = []
    print ('\n\t- Loading frames:')
    for i in range(0, len(X_list)):
        img = kImage.load_img(X_list[i])
        x = kImage.img_to_array(img)
        X.append(x)
        sys.stdout.write('\b' * len(str(i)))
        sys.stdout.write('\r')
        sys.stdout.write(str(i+1))
    
    del img, x, X_list
    X = np.asarray(X)
    print ('\nShape' + str(X.shape))
    
# For FgSegNet (multi-scale)
#    s2 = []
#    s3 = []
#    num_img = X.shape[0]
#    prev = 0
#    print ('\t- Downscale frames:')
#    for i in range(0, num_img):
#       pyramid = tuple(pyramid_gaussian(X[i]/255., max_layer=2, downscale=2))
#       s2.append(pyramid[1]*255.)
#       s3.append(pyramid[2]*255.)
#       sys.stdout.write('\b' * prev)
#       sys.stdout.write('\r')
#       s = str(i+1)
#       sys.stdout.write(s)
#       prev = len(s)
#       del pyramid
#    s2 = np.asarray(s2)
#    s3 = np.asarray(s3)
#    print ('\n')
#    print (s1.shape, s2.shape, s3.shape)

#    return [X, s2, s3] #return for FgSegNet (multi-scale)

    return X #return for FgSegNet_v2

def getData(ct_path, lung_mask_path, nodule_mask_path, dicom_path, folder):
    print('Data preparing...')

    ann = pd.read_csv(annotation, usecols=[0,1,2,3,5])
    ann = ann.values.tolist()

    X = []
    nodule_name = []
    local_list = []

    i = folder
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
            local_list.append((x, y, z))
            nodule_name.append(name)
            X.append(crop_nodule(ct_lung, x, y, z, crop_size))
    X = np.asarray(X).astype(np.float32)
    local_list = np.asarray(local_list)
    nodule_name = np.asarray(nodule_name)

    print('Data done')
                
    return [X, nodule_name, local_list]

# model dir
main_mdl_dir = os.path.join('FgSegNet_v2', 'Luna16', 'models')

# path to store results
results_dir = os.path.join('FgSegNet_v2', 'Luna16', 'results_ori')

if __name__ == '__main__':

    dicom_path = r'E:\NCKU\way\vnet.pytorch-master\luna16\luna16_ct'

    ct_path = r'E:\NCKU\Luna\npy\luna16_ct_spacing_npy'
    lung_mask_path = r'E:\NCKU\Luna\npy\LiDC_lung_mask_spacing_npy'
    nodule_mask_path = r'E:\NCKU\Luna\npy\LiDC_c3or_mask_spacing_npy'

    for i in range(folder_num):
        mdl_path = os.path.join(main_mdl_dir, 'mdl_' + 'folder' + str(i) + '.h5')
        mask_dir = os.path.join(results_dir, 'folder' + str(i))
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        # load model to segment
        model = load_model(mdl_path)

        data = getData(ct_path, lung_mask_path, nodule_mask_path, dicom_path, i)


# Loop through all categories (e.g. baseline)
for category, scene_list in dataset.items():
    # Loop through all scenes (e.g. highway, ...)
    for scene in scene_list:
        print ('\n->>> ' + category + ' / ' + scene)
        mdl_path = os.path.join(main_mdl_dir, category , 'mdl_' + scene + '.h5')
        
        mask_dir = os.path.join(results_dir, category, scene)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # path of dataset downloaded from CDNet
        scene_input_path = os.path.join(raw_dataset_dir, category, scene, 'input')
        # path of ROI to exclude non-ROI
        # make sure that each scene contains ROI.bmp and have the same dimension as raw RGB frames
        ROI_file = os.path.join(raw_dataset_dir, category, scene, 'ROI.bmp')
        
        # refer to http://jacarini.dinf.usherbrooke.ca/datasetOverview/
        img = kImage.load_img(ROI_file, grayscale=True)
        img = kImage.img_to_array(img)
        img = img.reshape(-1) # to 1D
        idx = np.where(img == 0.)[0] # get the non-ROI, black area
        del img
        
        # load path of files
        X_list = getFiles(scene_input_path)
        if (X_list is None):
            raise ValueError('X_list is None')

        # slice frames
        results = checkFrame(X_list)
        
        # load model to segment
        model = load_model(mdl_path)

        # if large numbers of frames, slice it
        if(results[0]): 
            for rangeee in results[1]: # for each slice
                slice_X_list =  X_list[rangeee]

                # load frames for each slice
                data = generateData(scene_input_path, slice_X_list, scene)
                
                # For FgSegNet (multi-scale only) 
                #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
                
                # For FgSegNet_v2
                Y_proba = model.predict(data, batch_size=1, verbose=1)
                del data

                # filter out
                shape = Y_proba.shape
                Y_proba = Y_proba.reshape([shape[0],-1])
                if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black
                        
                Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])

                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(slice_X_list[i]).replace('in','bin').replace('jpg','png')
                    x = Y_proba[i]
                    
#                    if batch_size in [2,4] and scene=='badminton':
#                        x = imresize(x, (480,720), interp='nearest')
#                        
#                    if batch_size in [2,4] and scene=='PETS2006':
#                        x = imresize(x, (576,720), interp='nearest')
                    
                    imsave(os.path.join(mask_dir, fname), x)
                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                    
                del Y_proba, slice_X_list

        else: # otherwise, no need to slice
            data = generateData(scene_input_path, X_list, scene)
            
            # For FgSegNet (multi-scale)
            #Y_proba = model.predict([data[0], data[1], data[2]], batch_size=batch_size, verbose=1) # (xxx, 240, 320, 1)
            
            # For FgSegNet_v2
            Y_proba = model.predict(data, batch_size=1, verbose=1)
            
            del data
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])
            if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black

            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])
            
            prev = 0
            print ('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(X_list[i]).replace('in','bin').replace('jpg','png')
                x = Y_proba[i]
                
#                if batch_size in [2,4] and scene=='badminton':
#                    x = imresize(x, (480,720), interp='nearest')
#                
#                if batch_size in [2,4] and scene=='PETS2006':
#                        x = imresize(x, (576,720), interp='nearest')
                        
                imsave(os.path.join(mask_dir, fname), x)
                sys.stdout.write('\b' * prev)
                sys.stdout.write('\r')
                s = str(i+1)
                sys.stdout.write(s)
                prev = len(s)
            del Y_proba
        del model, X_list, results

    gc.collect()