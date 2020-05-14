import os
import shutil
import json
import cv2
from IPython import embed
import numpy as np
LabelFolder = '../Images'
MetaDataFolder = '../MetaData'
JsonFolder = '../JsonData'
if not os.path.exists('{}'.format(JsonFolder)): os.mkdir('{}'.format(JsonFolder))
# COCO
# 'keypoints': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', \
# 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', \
# 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'], 

new_categories = [{'supercategory': 'person', 'id': 1, 'name': 'person', \
'keypoints': ['HeadTip', 'NoseTip', 'PupilLeft', 'PupilRight', 'MouthLeft', 'MouthRight', 'LeftEar', 'RightEar', \
'LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist', \
'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee', 'LeftAnkle', 'RightAnkle'], \
'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], \
[2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}] ### TODO

split = 'train'
image_id = 0 # Adjust according to the previous json
ann_id = 0 #
for batchfolder in os.listdir(MetaDataFolder):
	newFile = '{}/person_keypoints_{}_{}.json'.format(JsonFolder, batchfolder, split)
	new_json = {}
	new_json.update({'categories': new_categories})
	images_info = []
	anns_info = []

	for folder in os.listdir('{}/{}'.format(LabelFolder, batchfolder)):
		for image_path in os.listdir('{}/{}/{}'.format(LabelFolder, batchfolder, folder)):
			# images_info
			image = cv2.imread('{}/{}/{}/{}'.format(LabelFolder, batchfolder, folder, image_path))
			height, width = image.shape[:2]
			# ann_info
			json_path = '{}/{}/{}/{}.json'.format(MetaDataFolder, batchfolder, folder, image_path.split('.')[0])
			try:
				json_info = json.load(open(json_path,'r'))
			except:
				print('no json: ', json_path)
				continue
			if 'LabelInfo' not in json_info.keys():
				print('empty json: ', json_path)
				continue

			images_info.append({
				'file_name': '{}/{}/{}'.format(batchfolder, folder, image_path),
				'height':height,
				'width':width,
				'id':image_id
			})

			for label_name in json_info['LabelInfo']:
				label_info = json_info['LabelInfo'][label_name]
				label_info_keys = label_info.keys()
				if 'HeadLabelStatus' not in label_info_keys or 'BodyLabelStatus'not in label_info_keys or 'KeyPointsLabelStatus'not in label_info_keys:
					print('bug json: ', json_path)
					continue

				if label_info['HeadLabelStatus'] == 'True':
					bbox = label_info['HeadBoundingbox']
					hxmin,hymin,hxmax,hymax = int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])
					h_w, h_h = hxmax-hxmin, hymax-hymin
					have_head_box=1
				else:
					hxmin,hymin,h_w,h_h = [0,0,0,0]
					have_head_box=0

				if label_info['BodyLabelStatus'] == 'True':
					bbox = label_info['BodyBoundingbox']
					xmin,ymin,xmax,ymax = int(bbox['xmin']),  int(bbox['ymin']), int(bbox['xmax']),int(bbox['ymax'])
					w, h = xmax-xmin, ymax-ymin
					have_body_box = 1
				else:
					xmin,ymin,w,h = [0,0,0,0]
					have_body_box = 0

				keypoints_list = []
				if label_info['KeyPointsLabelStatus'] == 'True':
					for kp_name in new_categories[0]['keypoints']:
						visibility = label_info['KeyPoints'][kp_name][2]
						if visibility == 'Visible' :
							v=2
						elif visibility == 'Invisible' :
							v=1
						else:
							x,y,v = 0,0,0
						if v>0:
							x = int(label_info['KeyPoints'][kp_name][0])
							y = int(label_info['KeyPoints'][kp_name][1])
						keypoints_list.extend([x,y,v])
					have_kpt = 1
				else:
					keypoints_list = list(np.zeros((60))) ###
					have_kpt = 0
				num_keypoints = sum(kp > 0 for kp in keypoints_list[2::3])
				assert len(keypoints_list) == 60

				head_area = h_w*h_h
				body_area = w*h
				if body_area<head_area:
					anns_info.append({
						'HeadLabelStatus': have_body_box, 
						'HeadBoundingbox': [xmin, ymin, w, h],
						'BodyLabelStatus': have_head_box,
						'BodyBoundingbox': [hxmin,hymin,h_w,h_h],
						'KeyPointsLabelStatus': have_kpt, 
						'KeyPoints': keypoints_list, 
						'num_keypoints': int(num_keypoints), # must int
						'head_area': body_area,
						'body_area': head_area, 
						'iscrowd': 0,
						'image_id': image_id, 
						'bbox': [xmin, ymin, w, h],
						'category_id': 1, 
						'id': ann_id
					})
				else:
					anns_info.append({
						'HeadLabelStatus': have_head_box,
						'HeadBoundingbox': [hxmin,hymin,h_w,h_h],
						'BodyLabelStatus': have_body_box, 
						'BodyBoundingbox': [xmin, ymin, w, h],
						'KeyPointsLabelStatus': have_kpt, 
						'KeyPoints': keypoints_list, 
						'num_keypoints': int(num_keypoints), # must int
						'head_area': head_area,
						'body_area': body_area,
						'iscrowd': 0,
						'image_id': image_id, 
						'bbox': [xmin, ymin, w, h],
						'category_id': 1, 
						'id': ann_id
					})
				ann_id+=1

			image_id+=1

			if image_id %500==0:
				print(image_id)

	new_json.update({'images':images_info, 'annotations':anns_info})
	json.dump(new_json, open(newFile, 'w'))

	print('{} image_id:{}, ann_id:{}'.format(newFile, image_id, ann_id))
'''
json_info['LabelInfo']
{'9309d540-c396-4c21-b389-19ddc1b2d5a4': {'HeadLabelStatus': 'True',
  'HeadBoundingbox': {'xmin': 216, 'ymin': 28, 'xmax': 377, 'ymax': 257},
  'BodyLabelStatus': 'True',
  'BodyBoundingbox': {'xmin': 89, 'ymin': 8, 'xmax': 667, 'ymax': 439},
  'KeyPointsLabelStatus': 'True',
  'KeyPoints': {'PupilLeft': ['336', '150', 'Invisible'],
   'PupilRight': ['271', '155', 'Invisible'],
   'NoseTip': ['301', '188', 'Visible'],
   'MouthLeft': ['324', '206', 'Visible'],
   'MouthRight': ['271', '207', 'Visible'],
   'LeftEar': ['370', '170', 'Invisible'],
   'RightEar': ['228', '157', 'Invisible'],
   'HeadTip': ['298', '73', 'Invisible'],
   'LeftShoulder': ['432', '231', 'Visible'],
   'LeftElbow': ['585', '274', 'Visible'],
   'LeftWrist': ['628', '172', 'Visible'],
   'RightShoulder': ['162', '326', 'Visible'],
   'RightElbow': ['Invalid', 'Invalid', 'Untag'],
   'RightWrist': ['Invalid', 'Invalid', 'Untag'],
   'LeftHip': ['Invalid', 'Invalid', 'Untag'],
   'LeftKnee': ['Invalid', 'Invalid', 'Untag'],
   'LeftAnkle': ['Invalid', 'Invalid', 'Untag'],
   'RightHip': ['Invalid', 'Invalid', 'Untag'],
   'RightKnee': ['Invalid', 'Invalid', 'Untag'],
   'RightAnkle': ['Invalid', 'Invalid', 'Untag']}}}
'''

