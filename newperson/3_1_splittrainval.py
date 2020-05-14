import os
import shutil
import json
import cv2
from IPython import embed
import numpy as np
import glob
JsonFolder = '../JsonData'
if not os.path.exists('{}'.format(JsonFolder)): os.mkdir('{}'.format(JsonFolder))
split = 'trainval'
jsonFile = '{}/newperson_keypoints_{}.json'.format(JsonFolder, split)

valset_COCO_style = True

new_json_train = {}
new_json_val = {}
new_json_val_coco = {}
js = json.load(open(jsonFile, 'r'))
images_info = js['images']
anns_info = js['annotations']
new_json_train.update({'categories': js['categories']})
new_json_val.update({'categories': js['categories']})
new_json_val_coco.update({'categories': js['categories']})

new_json_train_img = []
new_json_val_img = []
new_json_train_ann = []
new_json_val_ann = []
new_json_val_ann_coco = []

for ann in anns_info:
	if ann['image_id']%5!=1:
		# train set 80%
		if images_info[ann['image_id']] not in new_json_train_img:
			new_json_train_img.append(images_info[ann['image_id']])
		new_json_train_ann.append(ann)
	else:
		# val set 20%
		if images_info[ann['image_id']] not in new_json_val_img:
			new_json_val_img.append(images_info[ann['image_id']])
		new_json_val_ann.append(ann)

		if valset_COCO_style:
			keypoints =ann['KeyPoints']
			del keypoints[12:18]
			del keypoints[0:3]
			num_keypoints = np.sum(vis > 0 for vis in np.array(keypoints)[2::3])

			new_json_val_ann_coco.append({
				'segmentation': ann['BodyBoundingbox'],
				'num_keypoints': int(num_keypoints), # must int
				'area': ann['body_area'],
				'iscrowd': 0,
				'keypoints': keypoints, 
				'image_id': ann['image_id'], 
				'bbox': ann['BodyBoundingbox'],
				'category_id': 1, 
				'id': ann['id']
			})


new_json_train.update({'images':new_json_train_img, 'annotations':new_json_train_ann})
new_json_val.update({'images':new_json_val_img, 'annotations':new_json_val_ann})

json.dump(new_json_train, open('{}/newperson_keypoints_train.json'.format(JsonFolder), 'w'))
json.dump(new_json_val, open('{}/newperson_keypoints_val.json'.format(JsonFolder), 'w'))

if valset_COCO_style:
	new_json_val_coco.update({'images':new_json_val_img, 'annotations':new_json_val_ann_coco})
	json.dump(new_json_val_coco, open('{}/newperson_keypoints_val_coco.json'.format(JsonFolder), 'w'))