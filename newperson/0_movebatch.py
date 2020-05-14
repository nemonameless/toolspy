import os
import shutil
import json
from IPython import embed
RootFolder = '../RawImages'
LabelFolder = '../Images'
NoLabelFolder = '../NoLabelImage'
MetaDataFolder = '../MetaData'

for batchfolder in os.listdir(RootFolder):
	for folder in os.listdir('{}/{}'.format(RootFolder, batchfolder)):
		tag_cnt = 0
		untag_cnt = 0
		for image_path in os.listdir('{}/{}/{}'.format(RootFolder, batchfolder, folder)):
			old_path = '{}/{}/{}/{}'.format(RootFolder, batchfolder, folder, image_path)
			
			jsonfile = '{}/{}/{}/{}.json'.format(MetaDataFolder, batchfolder, folder, image_path.split('.')[0])

			if not os.path.exists(jsonfile):
				new_path = '{}/{}/{}/{}'.format(NoLabelFolder, batchfolder, folder, image_path)
				untag_cnt+=1
				NoLabelBatch = '{}/{}'.format(NoLabelFolder, batchfolder)
				if not os.path.exists(NoLabelBatch):os.mkdir(NoLabelBatch)
				NoLabelBatchFolder = '{}/{}/{}'.format(NoLabelFolder, batchfolder, folder)
				if not os.path.exists(NoLabelBatchFolder):os.mkdir(NoLabelBatchFolder)
			else:
				new_path = '{}/{}/{}/{}'.format(LabelFolder, batchfolder, folder, image_path)
				tag_cnt+=1
				LabelBatch = '{}/{}'.format(LabelFolder, batchfolder)
				if not os.path.exists(LabelBatch):os.mkdir(LabelBatch)
				LabelBatchFolder = '{}/{}/{}'.format(LabelFolder, batchfolder, folder)
				if not os.path.exists(LabelBatchFolder):os.mkdir(LabelBatchFolder)
			shutil.copyfile(old_path, new_path)
		print('{}/{}: {} tag, {} untag.'.format(batchfolder, folder, tag_cnt, untag_cnt))
