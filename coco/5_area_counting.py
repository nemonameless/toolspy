import json
import numpy as np
from IPython import embed
jsonfile = '../JsonData/newperson_keypoints_trainval.json'
allannjson = json.load(open(jsonfile,'r'))
box0_32 =0
box32_64 = 0
box64_128 = 0
box128_256 = 0
box256_512 = 0
box512_1024 = 0
box1024_ = 0

valid_cnt=0
for i, ann in enumerate(allannjson['annotations']):
	if ann['iscrowd']:
		continue
	valid_cnt+=1
	x0,y0,w,h = ann['bbox'][:]
	if (w*h)**0.5 < 32:
		box0_32+=1
	elif 32 <= (w*h)**0.5 < 64:
		box32_64+=1
	elif 64 <= (w*h)**0.5 < 128:
		box64_128+=1
	elif 128 <= (w*h)**0.5 < 256:
		box128_256+=1
	elif 256 <= (w*h)**0.5 < 512:
		box256_512+=1
	elif 512 <= (w*h)**0.5 < 1024:
		box512_1024+=1
	elif 1024 <= (w*h)**0.5:
		box1024_+=1
print('all_ann: ', len(allannjson['annotations']))
print('valid_ann：', valid_cnt)
print('every cnt: {} | {} | {} | {} | {} | {} | {}'.format(box0_32,box32_64,box64_128,box128_256,box256_512,box512_1024,box1024_))
print(np.array([box0_32,box32_64,box64_128,box128_256,box256_512,box512_1024,box1024_])/valid_cnt)