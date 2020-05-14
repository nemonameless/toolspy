import glob
import json
import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
from 4_kmeans import kmeans, avg_iou
ANNOTATIONS_PATH = '../JsonData/newperson_keypoints_trainval.json'
CLUSTERS = 25
BBOX_NORMALIZE = True
#BBOX_NORMALIZE = False

#Display bouding box's size distribution and anchor generated in scatter.
def show_cluster(data, cluster, max_points=2000): 
	if len(data) > max_points:
		idx = np.random.choice(len(data), max_points)
		data = data[idx]
	plt.scatter(data[:,0], data[:,1], s=5, c='lavender')
	plt.scatter(cluster[:,0], cluster[:, 1], c='red', s=100, marker="^")
	plt.xlabel("Width")
	plt.ylabel("Height")
	plt.title("Bounding and anchor distribution")
	plt.savefig("cluster.png")
	plt.show()

# Display bouding box distribution with histgram.
def show_width_height(data, bins=50):
	if data.dtype != np.float32:
		data = data.astype(np.float32)
	width = data[:, 0]
	height = data[:, 1]
	ratio = height / width
	area = np.sqrt(height * width)

	plt.figure()
	plt.subplot(221)
	plt.hist(width, bins=bins, color='green')
	plt.xlabel('width')
	plt.ylabel('number')
	plt.title('Distribution of Width')

	plt.subplot(222)
	plt.hist(height,bins=bins, color='blue')
	plt.xlabel('Height')
	plt.ylabel('number')
	plt.title('Distribution of Height')

	plt.subplot(223)
	plt.hist(ratio, bins=bins,  color='magenta')
	plt.xlabel('Height / Width')
	plt.ylabel('number')
	plt.title('Distribution of aspect ratio(Height / Width)')

	plt.subplot(224)
	plt.hist(area, bins=bins,  color='red')
	plt.xlabel('sqrt(Height * Width')
	plt.ylabel('number')
	plt.title('Distribution of area(sqrt(Height * Width))')

	plt.savefig("shape-distribution.png")
	plt.show()
	
#Sort the cluster to with area small to big.
def sort_cluster(cluster):
	if cluster.dtype != np.float32:
		cluster = cluster.astype(np.float32)
	area = cluster[:, 0] * cluster[:, 1]
	cluster = cluster[area.argsort()]
	ratio = cluster[:,1:2] / cluster[:, 0:1]
	return np.concatenate([cluster, ratio], axis=-1)

def load_dataset_widerface_XML(path, normalized=True):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			if normalized:
				xmin = int(obj.findtext("bndbox/xmin")) / float(width)
				ymin = int(obj.findtext("bndbox/ymin")) / float(height)
				xmax = int(obj.findtext("bndbox/xmax")) / float(width)
				ymax = int(obj.findtext("bndbox/ymax")) / float(height)
			else:
				xmin = int(obj.findtext("bndbox/xmin")) 
				ymin = int(obj.findtext("bndbox/ymin")) 
				xmax = int(obj.findtext("bndbox/xmax")) 
				ymax = int(obj.findtext("bndbox/ymax"))
			if (xmax - xmin) == 0 or (ymax - ymin) == 0:
				continue # to avoid divded by zero error.
			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

def load_dataset_coco_json(path, normalized=True):
	gt = json.load(open(path, 'r'))
	images = {}
	for entry in gt['images']:
		images[entry['id']] = {
			'file_name': entry['file_name'],
			'width': entry['width'],
			'height': entry['height']
		}
	records = {}
	for ann in gt['annotations']:
		if ann['iscrowd'] > 0:
			continue
		if 'ignore' in ann and ann['ignore'] == 1:
			continue
		image_id = ann['image_id']
		if image_id not in records:
			records[image_id] = {}
		width, height = images[image_id]['width'], images[image_id]['height']
		# head bbox
		head_bbox = ann['HeadBoundingbox']
		head_bbox[2] += head_bbox[0]
		head_bbox[3] += head_bbox[1]
		# clip bbox to image
		head_bbox[0] = min(max(head_bbox[0], 0), width - 1)
		head_bbox[1] = min(max(head_bbox[1], 0), height - 1)
		head_bbox[2] = min(max(head_bbox[2], 0), width - 1)
		head_bbox[3] = min(max(head_bbox[3], 0), height - 1)
		if head_bbox[2] <= head_bbox[0] or head_bbox[3] <= head_bbox[1]:
			continue
		# body bbox
		body_bbox = ann['BodyBoundingbox']
		body_bbox[2] += body_bbox[0]
		body_bbox[3] += body_bbox[1]
		# clip bbox to image
		body_bbox[0] = min(max(body_bbox[0], 0), width - 1)
		body_bbox[1] = min(max(body_bbox[1], 0), height - 1)
		body_bbox[2] = min(max(body_bbox[2], 0), width - 1)
		body_bbox[3] = min(max(body_bbox[3], 0), height - 1)
		if body_bbox[2] <= body_bbox[0] or body_bbox[3] <= body_bbox[1]:
			continue

		head_bbox_flag = 1 if ann['HeadLabelStatus'] == 'True' else 0
		body_bbox_flag = 1 if ann['BodyLabelStatus'] == 'True' else 0
		keypoint_flag = 1 if ann['KeyPointsLabelStatus'] == 'True' else 0

		records[image_id].setdefault('HeadBoundingbox', []).append(head_bbox)
		records[image_id].setdefault('HeadLabelStatus', []).append(head_bbox_flag)
		records[image_id].setdefault('BodyBoundingbox', []).append(body_bbox)
		records[image_id].setdefault('BodyLabelStatus', []).append(body_bbox_flag)

		records[image_id].setdefault('num_keypoints', []).append(ann["num_keypoints"])
		records[image_id].setdefault('KeyPointsLabelStatus', []).append(keypoint_flag)
		#records[image_id].setdefault('keypoints', []).append(ann['keypoints']) ###
		keypoints = ann["KeyPoints"]
		coords = np.vstack((keypoints[0::3], keypoints[1::3], keypoints[2::3])).transpose()
		records[image_id].setdefault('KeyPoints', []).append(coords)

		records[image_id].setdefault('body_area', []).append(ann['body_area'])
		records[image_id].setdefault('head_area', []).append(ann['head_area'])
		records[image_id].setdefault('iscrowd', []).append(ann['iscrowd'])
		records[image_id].setdefault('category_id', []).append(ann['category_id'])
		records[image_id].setdefault('ann_id', []).append(ann['id'])

	body_boxes_dataset = []
	head_boxes_dataset = []

	records_gt = []
	for image_id in images:
		record_t = {}
		record_t['image_id'] = image_id
		record_t['image_name'] = images[image_id]['file_name']
		record_t['width'] = images[image_id]['width']
		record_t['height'] = images[image_id]['height']

		if image_id in records:
			record_t.update(records[image_id])
			# record_t['body_area'] = numpy.asarray(record_t['body_area'], dtype=numpy.float32)
			# record_t['head_area'] = numpy.asarray(record_t['head_area'], dtype=numpy.float32)
			# record_t['iscrowd'] = numpy.asarray(record_t['iscrowd'], dtype=numpy.bool)
			# record_t['category_id'] = numpy.asarray(record_t['category_id'], dtype=numpy.long)
			# record_t['HeadLabelStatus'] = numpy.asarray(record_t['HeadLabelStatus'], dtype=numpy.long)
			# record_t['BodyLabelStatus'] = numpy.asarray(record_t['BodyLabelStatus'], dtype=numpy.long)
			# record_t['KeyPointsLabelStatus'] = numpy.asarray(record_t['KeyPointsLabelStatus'], dtype=numpy.long)
			#record_t['HeadBoundingbox'] = numpy.asarray(record_t['HeadBoundingbox'], dtype=numpy.float32)
			#record_t['BodyBoundingbox'] = numpy.asarray(record_t['BodyBoundingbox'], dtype=numpy.float32)
			# record_t['KeyPoints'] = numpy.asarray(record_t['KeyPoints'], dtype=numpy.long)
			# record_t['num_keypoints'] = numpy.asarray(record_t['num_keypoints'], dtype=numpy.long)
			if not 'BodyBoundingbox' in record_t.keys():
				continue
			for bbox in record_t['BodyBoundingbox']:
				if normalized:
					xmin = int(bbox[0]) / float(record_t['width'])
					ymin = int(bbox[1]) / float(record_t['height'])
					xmax = int(bbox[2]) / float(record_t['width'])
					ymax = int(bbox[3]) / float(record_t['height'])
				else:
					xmin = int(bbox[0])
					ymin = int(bbox[1])
					xmax = int(bbox[2])
					ymax = int(bbox[3]) 
				if (xmax - xmin) == 0 or (ymax - ymin) == 0:
					continue # to avoid divded by zero error.
				body_boxes_dataset.append([xmax - xmin, ymax - ymin])

	return np.array(body_boxes_dataset)

print("Start to load data annotations on: %s" % ANNOTATIONS_PATH)
data = load_dataset_coco_json(ANNOTATIONS_PATH, normalized=BBOX_NORMALIZE)

### kmeans
print("Start to do kmeans, please wait for a moment.")
out = kmeans(data, k=CLUSTERS)
out_sorted = sort_cluster(out)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
show_cluster(data, out, max_points=2000)
if out.dtype != np.float32:
	out = out.astype(np.float32)
print("Recommanded aspect ratios(width/height)")
print("Width    Height   Height/Width")
for i in range(len(out_sorted)):
	print("%.3f      %.3f     %.1f" % (out_sorted[i,0], out_sorted[i,1], out_sorted[i,2]))

show_width_height(data, bins=50)