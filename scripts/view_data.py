import numpy as np
import pandas as pd
import os
import cv2
import SimpleITK as sitk

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

	ann = pd.read_csv(annotation, usecols=[0,1,2,3,5])
	ann = ann.values.tolist()

	# class_ann = pd.read_csv(class_annotation, usecols=[0,5])
	# local_ann = pd.read_csv(local_annotation, usecols=[0,1,2,3])
	# class_ann = class_ann.values.tolist()
	# local_ann = local_ann.values.tolist()

	# for i in class_ann:
	# 	i[0] = i[0].split('-')[0]

	X = []
	Y = []
	C = []
	for i in range(folder_num):
		if i == folder:
			continue
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
			nodule_mask[nodule_mask!=0] = 255

			for d in idx:
				x = ann[d][1]
				y = ann[d][2]
				z = ann[d][3]

				C.append(ann[d][4])
				X.append(crop_nodule(ct_lung, x, y, z, crop_size))
				Y.append(crop_nodule(nodule_mask, x, y, z, crop_size))
				
	return [X, Y, C]



if __name__ == '__main__':
	ct_path = r'E:\NCKU\Luna\npy\luna16_ct_spacing_npy'
	lung_mask_path = r'E:\NCKU\Luna\npy\LiDC_lung_mask_spacing_npy'
	nodule_mask_path = r'E:\NCKU\Luna\npy\LiDC_c3or_mask_spacing_npy'

	dicom_path = r'E:\NCKU\way\vnet.pytorch-master\luna16\luna16_ct'

	for i in range(folder_num):
		getData(ct_path, lung_mask_path, nodule_mask_path, dicom_path, i)

