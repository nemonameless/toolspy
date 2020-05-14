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

newFile = '{}/newperson_keypoints_{}.json'.format(JsonFolder, split)
new_json = {}
new_categories = [{'supercategory': 'person', 'id': 1, 'name': 'person', \
'keypoints': ['HeadTip', 'NoseTip', 'PupilLeft', 'PupilRight', 'MouthLeft', 'MouthRight', 'LeftEar', 'RightEar', \
'LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow', 'LeftWrist', 'RightWrist', \
'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee', 'LeftAnkle', 'RightAnkle'], \
'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], \
[2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]}] ### TODO
new_json.update({'categories': new_categories})

images_info = []
anns_info = []
alljsons = glob.glob('{}/*.json'.format(JsonFolder))
alljsons.sort()
for json_file in alljsons:
	js = json.load(open(json_file, 'r'))
	images_info.extend(js['images'])
	anns_info.extend(js['annotations'])
	print('json_file:{}'.format(json_file))

new_json.update({'images':images_info, 'annotations':anns_info})
json.dump(new_json, open(newFile, 'w'))
