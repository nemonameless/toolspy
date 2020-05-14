import os
import shutil
import json
import cv2
from IPython import embed
import numpy as np
MetaDataFolder = '../MetaData'
MetaDataCorrectFolder = '../MetaDataCorrect'
MetaDataBugFolder = '../MetaDataBug'
if not os.path.exists('{}'.format(MetaDataCorrectFolder)): os.mkdir('{}'.format(MetaDataCorrectFolder))
if not os.path.exists('{}'.format(MetaDataBugFolder)): os.mkdir('{}'.format(MetaDataBugFolder))

bug_json_cnt=0
for batchfolder in os.listdir(MetaDataFolder):
	if not os.path.exists('{}/{}'.format(MetaDataCorrectFolder, batchfolder)): 
		os.mkdir('{}/{}'.format(MetaDataCorrectFolder, batchfolder))
	if not os.path.exists('{}/{}'.format(MetaDataBugFolder, batchfolder)): 
		os.mkdir('{}/{}'.format(MetaDataBugFolder, batchfolder))

	for folder in os.listdir('{}/{}'.format(MetaDataFolder, batchfolder)):
		if not os.path.isdir('{}/{}/{}'.format(MetaDataFolder, batchfolder, folder)):
			continue
		if not os.path.exists('{}/{}/{}'.format(MetaDataCorrectFolder, batchfolder, folder)): 
			os.mkdir('{}/{}/{}'.format(MetaDataCorrectFolder, batchfolder, folder))
		if not os.path.exists('{}/{}/{}'.format(MetaDataBugFolder, batchfolder, folder)): 
			os.mkdir('{}/{}/{}'.format(MetaDataBugFolder, batchfolder, folder))

		for json_path in os.listdir('{}/{}/{}'.format(MetaDataFolder, batchfolder, folder)):
			json_fullpath = '{}/{}/{}/{}'.format(MetaDataFolder, batchfolder, folder, json_path)
			json_info = json.load(open(json_fullpath,'r'))
			# new json
			new_json = json_info
			new_json_path = '{}/{}/{}/{}'.format(MetaDataCorrectFolder, batchfolder, folder, json_path)

			bug_flag=0
			for label_name in json_info['LabelInfo']:
				label_info = json_info['LabelInfo'][label_name]

				hbox = label_info['HeadBoundingbox']
				if hbox == {}:
					continue
				hxmin,hymin,hxmax,hymax = int(hbox['xmin']), int(hbox['ymin']), int(hbox['xmax']), int(hbox['ymax'])
				h_w, h_h = hxmax-hxmin, hymax-hymin

				bbox = label_info['BodyBoundingbox']
				if bbox == {}:
					continue
				xmin,ymin,xmax,ymax = int(bbox['xmin']),  int(bbox['ymin']), int(bbox['xmax']),int(bbox['ymax'])
				w, h = xmax-xmin, ymax-ymin

				head_area = h_w*h_h
				body_area = w*h

				if body_area<head_area:
					bug_flag=1
					new_json['LabelInfo'][label_name]['HeadBoundingbox'] = bbox
					new_json['LabelInfo'][label_name]['BodyBoundingbox'] = hbox
				else:
					continue

			if bug_flag:
				bug_json_cnt+=1
				bug_json_path = '{}/{}/{}/{}'.format(MetaDataBugFolder, batchfolder, folder, json_path)
				shutil.copyfile(json_fullpath, bug_json_path)

				json.dump(new_json, open(new_json_path, 'w'),indent=4)
			else:
				shutil.copyfile(json_fullpath, new_json_path)

		print('{}/{}, bug json cnt:{}'.format(batchfolder, folder, bug_json_cnt))