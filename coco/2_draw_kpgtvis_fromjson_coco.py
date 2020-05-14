import os
import json
import random
import numpy
from IPython import embed
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2 as cv
JsonFolder = './annotations'
ImageFolder = 'valdetvis_coco05'

class keyPointVisDataset(object):
    def __init__(self, split):
        self.split = split
        self.gt_file = '{}/human_bboxkp_{}2017.json'.format(JsonFolder, self.split)

    def load_bbox_keypoint_vis_gt(self, atype='keypoints'):
        gt = json.load(open(self.gt_file, 'r'))
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
             # body bbox
            body_bbox = ann['bbox']
            body_bbox[2] += body_bbox[0]
            body_bbox[3] += body_bbox[1]
            # clip bbox to image
            body_bbox[0] = min(max(body_bbox[0], 0), width - 1)
            body_bbox[1] = min(max(body_bbox[1], 0), height - 1)
            body_bbox[2] = min(max(body_bbox[2], 0), width - 1)
            body_bbox[3] = min(max(body_bbox[3], 0), height - 1)
            #if body_bbox[2] <= body_bbox[0] or body_bbox[3] <= body_bbox[1]:
            #    continue
            records[image_id].setdefault('bbox', []).append(body_bbox)
            records[image_id].setdefault('num_keypoints', []).append(ann["num_keypoints"])
            keypoints = ann["keypoints"]
            coords = np.vstack((keypoints[0::3], keypoints[1::3], keypoints[2::3])).transpose()
            records[image_id].setdefault('keypoints', []).append(coords)
            records[image_id].setdefault('area', []).append(ann['area'])
            records[image_id].setdefault('iscrowd', []).append(ann['iscrowd'])
            records[image_id].setdefault('category_id', []).append(ann['category_id'])
            records[image_id].setdefault('id', []).append(ann['id'])


        records_gt = []
        for image_id in images:
            record_t = {}
            record_t['image_id'] = image_id
            record_t['image_name'] = images[image_id]['file_name']
            record_t['width'] = images[image_id]['width']
            record_t['height'] = images[image_id]['height']
            if image_id in records:
                record_t.update(records[image_id])
                record_t['area'] = numpy.asarray(record_t['area'], dtype=numpy.float32)
                record_t['iscrowd'] = numpy.asarray(record_t['iscrowd'], dtype=numpy.bool)
                record_t['category_id'] = numpy.asarray(record_t['category_id'], dtype=numpy.long)
                record_t['bbox'] = numpy.asarray(record_t['bbox'], dtype=numpy.float32)
                record_t['keypoints'] = numpy.asarray(record_t['keypoints'], dtype=numpy.long)
                record_t['num_keypoints'] = numpy.asarray(record_t['num_keypoints'], dtype=numpy.long)
            else:
                continue
            records_gt.append(record_t)

        return records_gt

    def extract_image(self, record):
        file_path = '{}/{}'.format(ImageFolder, record['image_name'])
        image = Image.open(file_path).convert('RGB')
        return image

    def transform(self, record):
        record['image'] = self.extract_image(record)
        return record

def map_coco_to_personlab(keypoints):
    permute = [0, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    return keypoints[:, permute, :]
def plot_poses(img, skeletons, save_name='pose.jpg'):
    EDGES = [
        (0, 14),
        (0, 13),
        (0, 4),
        (0, 1),
        (14, 16),
        (13, 15),
        (4, 10),
        (1, 7),
        (10, 11),
        (7, 8),
        (11, 12),
        (8, 9),
        (4, 5),
        (1, 2),
        (5, 6),
        (2, 3)
    ]
    NUM_EDGES = len(EDGES)
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    #plt.figure()
    canvas = img.copy()

    for i in range(17):
        for j in range(len(skeletons)):
            #cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 2, colors[i], thickness=-1)
            if skeletons[j][i][2]==0:
                cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, [255, 0, 0], thickness=-1) # green
            elif skeletons[j][i][2]==1:
                cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, [255, 0, 0], thickness=-1) # blue
            else:
                cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, [255, 0, 0], thickness=-1) # red

    to_plot = cv.addWeighted(img, 0.3, canvas, 0.7, 0)
    #fig = matplotlib.pyplot.gcf()
    # stickwidth = 2
    # skeletons = map_coco_to_personlab(skeletons)
    # for i in range(NUM_EDGES):
    #     for j in range(len(skeletons)):
    #         edge = EDGES[i]
    #         if skeletons[j][edge[0],2] == 0 or skeletons[j][edge[1],2] == 0:
    #             continue
    #         cur_canvas = canvas.copy()
    #         X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
    #         Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
    #         mX = np.mean(X)
    #         mY = np.mean(Y)
    #         length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
    #         angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
    #         polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
    #         cv.fillConvexPoly(cur_canvas, polygon, (0,0,0))#colors[i])
    #         canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

def display_gt_keypoint(vis_one_image, image, body_boxes, keypoints, labels=[], save_name=None):
    if len(body_boxes) == 0:
        return
    if len(keypoints) > 0:
        assert(len(body_boxes) == len(keypoints))
    if len(labels) > 0:
        assert(len(body_boxes) == len(labels))
    image = np.array(image)

    for idx, bbox in enumerate(body_boxes):
        colors_gt = (255, 0, 0)
        image = cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors_gt, 1)
        #handle.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=colors_gt, linewidth=2))

    image = plot_poses(image, np.array(keypoints))

    image = Image.fromarray(image)
    if not vis_one_image:
        image.save(save_name)
    else:
        _, handle = plt.subplots(figsize=(12, 12))
        handle.imshow(image, aspect='equal')
        handle.set_title('%d gt_boxes'%(len(body_boxes)), fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    VisDataFolder = './valdetgtvis'
    split = 'val'
    #vis_one_image = True
    vis_one_image = False

    if not vis_one_image:
        if not os.path.exists(VisDataFolder): 
            os.mkdir(VisDataFolder)

    kpts = keyPointVisDataset(split)
    for rec in kpts.load_bbox_keypoint_vis_gt(): 
        if not vis_one_image:
            save_name = '{}/{}'.format(VisDataFolder, rec['image_name'])
        else:
            save_name=None
        image = kpts.extract_image(rec)
        #print(rec['image_name'])
        display_gt_keypoint(vis_one_image, image, rec['bbox'], rec['keypoints'], rec['category_id'], save_name)
####
# import json
# a=json.load(open('human_train2017.json','r'))
# b=json.load(open('person_keypoints_train2017.json','r'))
# newFile = 'human_bboxkp_train2017.json'
# new_json={}
# new_json.update({'images':a['images'], 'annotations':b['annotations'], 'categories':a['categories']})
# json.dump(new_json, open(newFile, 'w'))
