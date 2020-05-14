from mmdet.apis import init_detector, inference_detector, show_result, show_result_keypoint_bbox_head
import mmcv
import os
import json
config_file = '111.py'
checkpoint_file = './work_newperson/111/epoch_24.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

vis_from_folder = 0

if vis_from_folder:
	images_root_path = './Bodybuilding'
	images_save_path = './vis'
	if not os.path.exists(images_save_path):
		os.mkdir(images_save_path)
	for i, img in enumerate(os.listdir(images_root_path)):
		img_path = '{}/{}'.format(images_root_path,img)
		out_file_path = '{}/{}'.format(images_save_path,img)
		result = inference_detector(model, img_path)
		show_result_keypoint_bbox_head(img_path, result, model.CLASSES, score_thr=0.5, wait_time=0, show=False, out_file=out_file_path)
		if i%10==0:
			print(i)
else:
	images_root_path = './data/coco/Data/ImagesGTvis/'
	images_save_path = './visgtdet05'
	json_file = './data/coco/Data/JsonData/111.json'
	json_info = json.load(open(json_file,'r'))
	if not os.path.exists(images_save_path):
		os.mkdir(images_save_path)
	for i, image in enumerate(json_info['images']):
		img=image['file_name']
		img_path = '{}/{}'.format(images_root_path,img)
		out_img = img.replace(' ','_').replace('/','_')
		out_file_path = '{}/{}'.format(images_save_path,out_img)
		result = inference_detector(model, img_path)
		show_result_keypoint_bbox_head(img_path, result, model.CLASSES, score_thr=0.5, wait_time=0, show=False, out_file=out_file_path)
		if i%100==0:
			print(i)