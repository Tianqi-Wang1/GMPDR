import numpy as np
import pandas as pd
import os
import cv2
from copy import deepcopy
import random

def rot(df, image, k=1):
	'''
	Rotates image and coordinates counter-clockwise by k * 90° within image origin
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:param k: Number of times to rotate by 90°
	:return: Rotated Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	c, s = np.cos(-k * np.pi / 2), np.sin(-k * np.pi / 2)
	R = np.array([[c, s], [-s, c]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	for i in range(k):
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def fliplr(df, image):
	'''
	Flip image and coordinates horizontally
	:param df: Pandas DataFrame with at least columns 'x' and 'y'
	:param image: PIL Image
	:return: Flipped Dataframe and image
	'''
	xy = df.copy()
	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] - x0 / 2
	xy.loc()[:, 'y'] = xy['y'] - y0 / 2
	R = np.array([[-1, 0], [0, 1]])
	xy.loc()[:, ['x', 'y']] = np.dot(xy[['x', 'y']], R)
	image = cv2.flip(image, 1)

	if image.ndim == 3:
		y0, x0, channels = image.shape
	else:
		y0, x0= image.shape

	xy.loc()[:, 'x'] = xy['x'] + x0 / 2
	xy.loc()[:, 'y'] = xy['y'] + y0 / 2
	return xy, image


def augment_data(data, image_path='data/SDD/train', image_file='reference.jpg', seg_mask=False):
	images = {}
	ks = [1, 2, 3]
	for scene in data.sceneId.unique():
		im_path = os.path.join(image_path, scene, image_file)
		if seg_mask:
			im = cv2.imread(im_path, 0)
		else:
			im = cv2.imread(im_path)
		images[scene] = im
	data_ = data.copy()  # data without rotation, used so rotated data can be appended to original df
	k2rot = {1: '_rot90', 2: '_rot180', 3: '_rot270'}
	for k in ks:
		metaId_max = data['metaId'].max()
		for scene in data_.sceneId.unique():
			im_path = os.path.join(image_path, scene, image_file)
			if seg_mask:
				im = cv2.imread(im_path, 0)
			else:
				im = cv2.imread(im_path)

			data_rot, im = rot(data_[data_.sceneId == scene], im, k)
			# image
			rot_angle = k2rot[k]
			images[scene + rot_angle] = im

			data_rot['sceneId'] = scene + rot_angle
			data_rot['metaId'] = data_rot['metaId'] + metaId_max + 1
			# data = data.append(data_rot)
			data = pd.concat([data, data_rot], ignore_index=True)

	metaId_max = data['metaId'].max()
	for scene in data.sceneId.unique():
		im = images[scene]
		data_flip, im_flip = fliplr(data[data.sceneId == scene], im)
		data_flip['sceneId'] = data_flip['sceneId'] + '_fliplr'
		data_flip['metaId'] = data_flip['metaId'] + metaId_max + 1
		# data = data.append(data_flip)
		data = pd.concat([data, data_flip], ignore_index=True)
		images[scene + '_fliplr'] = im_flip

	return data, images

def mask_trajectory(df, mask_ratio=0.3, mask_value=np.nan, mask_steps=8):
    df_masked = df.copy()
    for meta_id in df_masked['metaId'].unique():
        traj = df_masked[df_masked['metaId'] == meta_id].copy()
        traj_mask_region = traj.iloc[:mask_steps]
        mask_candidates = traj_mask_region.index.tolist()
        
        n_mask = int(len(mask_candidates) * mask_ratio)
        if n_mask > 0:
            masked_idxs = random.sample(mask_candidates, n_mask)
            df_masked.loc[masked_idxs, ['x', 'y']] = mask_value

    return df_masked



def augment_data_random(data, image_path='data/SDD/train', image_file='reference.jpg', seg_mask=False, n_aug=1):
	images = {}
	for scene in data.sceneId.unique():
		im_path = os.path.join(image_path, scene, image_file)
		im = cv2.imread(im_path, 0) if seg_mask else cv2.imread(im_path)
		images[scene] = im

	data_original = data.copy()
	metaId_max = data_original['metaId'].max()

	offset = metaId_max + 1

	augmented_data_list = []
	images_augmented = {}

	rotation_options = [0, 90, 180, 270]
	apply_masking = True

	for i in range(n_aug):
		rot_angle = random.choice(rotation_options)
		flip_decision = random.choice([True, False])

		for scene in data_original.sceneId.unique():
			im = images[scene]
			data_scene = data_original[data_original.sceneId == scene].copy()

			if rot_angle != 0:
				k = rot_angle // 90
				data_scene, im = rot(data_scene, im, k)
			if flip_decision:
				data_scene, im = fliplr(data_scene, im)

			if apply_masking:
				data_scene = mask_trajectory(data_scene, mask_ratio=0.3)

			scene_aug_name = f"{scene}_aug{i}_rot{rot_angle}" + ("_fliplr" if flip_decision else "")
			data_scene['sceneId'] = scene_aug_name
			data_scene['metaId'] = data_scene['metaId'] + offset
			augmented_data_list.append(data_scene)
			images_augmented[scene_aug_name] = im

		offset += metaId_max + 1

	augmented_data = pd.concat(augmented_data_list, ignore_index=True)
	return augmented_data, images_augmented


def augment_data_select(data, image_path='data/SDD/train', image_file='reference.jpg', seg_mask=False):
	images = {}
	for scene in data.sceneId.unique():
		im_path = os.path.join(image_path, scene, image_file)
		im = cv2.imread(im_path, 0) if seg_mask else cv2.imread(im_path)
		images[scene] = im

	data_original = data.copy()
	metaId_max = data_original['metaId'].max()
	offset = metaId_max + 1

	augmented_data_list = []
	images_augmented = {}

	rotation_options = [0]

	for i in range(1):
		rot_angle = random.choice(rotation_options)
		# flip_decision = random.choice([True, False])

		for scene in data_original.sceneId.unique():
			im = images[scene]
			data_scene = data_original[data_original.sceneId == scene].copy()

			if rot_angle != 0:
				k = rot_angle // 90
				data_scene, im = rot(data_scene, im, k)
			# if flip_decision:
			# 	data_scene, im = fliplr(data_scene, im)

			scene_aug_name = f"{scene}_aug{i}_rot{rot_angle}"
			data_scene['sceneId'] = scene_aug_name
			data_scene['metaId'] = data_scene['metaId'] + offset
			augmented_data_list.append(data_scene)
			images_augmented[scene_aug_name] = im

		offset += metaId_max + 1

	augmented_data = pd.concat(augmented_data_list, ignore_index=True)
	return augmented_data, images_augmented

def augment_data_squence(data, image_path='data/SDD/train', image_file='reference.jpg', seg_mask=False, n_aug=0):
	images = {}
	for scene in data.sceneId.unique():
		im_path = os.path.join(image_path, scene, image_file)
		im = cv2.imread(im_path, 0) if seg_mask else cv2.imread(im_path)
		images[scene] = im

	data_original = data.copy()
	metaId_max = data_original['metaId'].max()
	offset = metaId_max + 1

	augmented_data_list = []
	images_augmented = {}

	rotation_options = [0, 90, 180, 270]
	flip_options = [True, False]

	r = n_aug % 4
	f = n_aug // 4
	rot_angle = rotation_options[r]
	flip_decision = flip_options[f]
	apply_masking = True

	for scene in data_original.sceneId.unique():
		im = images[scene]
		data_scene = data_original[data_original.sceneId == scene].copy()

		if rot_angle != 0:
			k = rot_angle // 90
			data_scene, im = rot(data_scene, im, k)
		if flip_decision:
			data_scene, im = fliplr(data_scene, im)

		if apply_masking:
			data_scene = mask_trajectory(data_scene, mask_ratio=0.3)

		scene_aug_name = f"{scene}_aug{n_aug}_rot{rot_angle}" + ("_fliplr" if flip_decision else "")
		data_scene['sceneId'] = scene_aug_name
		data_scene['metaId'] = data_scene['metaId'] + offset
		augmented_data_list.append(data_scene)
		images_augmented[scene_aug_name] = im

	augmented_data = pd.concat(augmented_data_list, ignore_index=True)
	return augmented_data, images_augmented

def create_images_dict(data, image_path, image_file='reference.jpg'):
	images = {}
	for scene in data.sceneId.unique():
		if image_file == 'oracle.png':
			im = cv2.imread(os.path.join(image_path, scene, image_file), 0)
		else:
			im = cv2.imread(os.path.join(image_path, scene, image_file))
		images[scene] = im
	return images