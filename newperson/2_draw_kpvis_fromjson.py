import os
import json
import random
import numpy
import torch
from IPython import embed
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2 as cv
import cv2
JsonFolder = '../JsonData'
if not os.path.exists('{}'.format(JsonFolder)): os.mkdir('{}'.format(JsonFolder))

class keyPointVisDataset(object):
    def __init__(self, batch, split):
        self.batch = batch
        self.split = split
        self.gt_file = '{}/person_keypoints_{}_{}.json'.format(JsonFolder, self.batch, self.split)

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
            # head bbox
            head_bbox = ann['HeadBoundingbox']
            head_bbox[2] += head_bbox[0]
            head_bbox[3] += head_bbox[1]
            # clip bbox to image
            head_bbox[0] = min(max(head_bbox[0], 0), width - 1)
            head_bbox[1] = min(max(head_bbox[1], 0), height - 1)
            head_bbox[2] = min(max(head_bbox[2], 0), width - 1)
            head_bbox[3] = min(max(head_bbox[3], 0), height - 1)
            #if head_bbox[2] <= head_bbox[0] or head_bbox[3] <= head_bbox[1]:
            #    continue
             # body bbox
            body_bbox = ann['BodyBoundingbox']
            body_bbox[2] += body_bbox[0]
            body_bbox[3] += body_bbox[1]
            # clip bbox to image
            body_bbox[0] = min(max(body_bbox[0], 0), width - 1)
            body_bbox[1] = min(max(body_bbox[1], 0), height - 1)
            body_bbox[2] = min(max(body_bbox[2], 0), width - 1)
            body_bbox[3] = min(max(body_bbox[3], 0), height - 1)
            #if body_bbox[2] <= body_bbox[0] or body_bbox[3] <= body_bbox[1]:
            #    continue

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


        records_gt = []
        for image_id in images:
            record_t = {}
            record_t['image_id'] = image_id
            record_t['image_name'] = images[image_id]['file_name']
            record_t['width'] = images[image_id]['width']
            record_t['height'] = images[image_id]['height']

            if image_id in records:
                record_t.update(records[image_id])
                record_t['body_area'] = numpy.asarray(record_t['body_area'], dtype=numpy.float32)
                record_t['head_area'] = numpy.asarray(record_t['head_area'], dtype=numpy.float32)
                record_t['iscrowd'] = numpy.asarray(record_t['iscrowd'], dtype=numpy.bool)
                record_t['category_id'] = numpy.asarray(record_t['category_id'], dtype=numpy.long)
                record_t['HeadLabelStatus'] = numpy.asarray(record_t['HeadLabelStatus'], dtype=numpy.long)
                record_t['BodyLabelStatus'] = numpy.asarray(record_t['BodyLabelStatus'], dtype=numpy.long)
                record_t['KeyPointsLabelStatus'] = numpy.asarray(record_t['KeyPointsLabelStatus'], dtype=numpy.long)

                record_t['HeadBoundingbox'] = numpy.asarray(record_t['HeadBoundingbox'], dtype=numpy.float32)
                record_t['BodyBoundingbox'] = numpy.asarray(record_t['BodyBoundingbox'], dtype=numpy.float32)
                record_t['KeyPoints'] = numpy.asarray(record_t['KeyPoints'], dtype=numpy.long)
                record_t['num_keypoints'] = numpy.asarray(record_t['num_keypoints'], dtype=numpy.long)
            else:
                continue
            records_gt.append(record_t)

        return records_gt

    def extract_image(self, record):
        file_path = '{}/{}'.format(LabelFolder, record['image_name'])
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
    h,w = img.shape[0], img.shape[1]


    left_vis_color = [255,0,0] # red
    right_vis_color =  [128,42,42] 
    middle_vis_color = [188,143,143] 

    left_invis_color = [0, 0, 255] 
    right_invis_color = [8,46,84]
    middle_invis_color = [64,224,205]


    basex, basey = int(w/40), int(h/32)
    text_start = int(w/40*1.5)
    text_h = int(h/32)
    cv.circle(canvas, (basex, basey+text_h*0), 4, left_vis_color, thickness=-1)
    cv.circle(canvas, (basex, basey+text_h*1), 4, right_vis_color, thickness=-1)
    cv.circle(canvas, (basex, basey+text_h*2), 4, middle_vis_color, thickness=-1)

    cv.circle(canvas, (basex, basey+text_h*3), 4, left_invis_color, thickness=-1)
    cv.circle(canvas, (basex, basey+text_h*4), 4, right_invis_color, thickness=-1)
    cv.circle(canvas, (basex, basey+text_h*5), 4, middle_invis_color, thickness=-1)

    cv.circle(canvas, (basex, basey+text_h*6), 4, (255,128,0), thickness=-1)
    cv.circle(canvas, (basex, basey+text_h*7), 4, (255, 255, 0), thickness=-1)
    basey = int(basey*1.3)
    img = cv2.putText(canvas, 'Left Vis', (text_start, basey+text_h*0), cv.FONT_HERSHEY_SIMPLEX, 0.4, left_vis_color, 1)
    img = cv2.putText(canvas, 'Right Vis', (text_start, basey+text_h*1), cv.FONT_HERSHEY_SIMPLEX, 0.4, right_vis_color, 1)
    img = cv2.putText(canvas, 'Middle Vis', (text_start, basey+text_h*2), cv.FONT_HERSHEY_SIMPLEX, 0.4, middle_vis_color, 1)

    img = cv2.putText(canvas, 'Left Invis', (text_start, basey+text_h*3), cv.FONT_HERSHEY_SIMPLEX, 0.4, left_invis_color, 1)
    img = cv2.putText(canvas, 'Right Invis', (text_start, basey+text_h*4), cv.FONT_HERSHEY_SIMPLEX, 0.4, right_invis_color, 1)
    img = cv2.putText(canvas, 'Middle Invis', (text_start, basey+text_h*5), cv.FONT_HERSHEY_SIMPLEX, 0.4, middle_invis_color, 1)

    img = cv2.putText(canvas, 'Head Bbox', (text_start, basey+text_h*6), cv.FONT_HERSHEY_SIMPLEX, 0.4, (210,180,140) , 1)# orange
    img = cv2.putText(canvas, 'Body Bbox', (text_start, basey+text_h*7), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)# yellow

    for i in range(20):
        for j in range(len(skeletons)):
            #cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 2, colors[i], thickness=-1)
            if skeletons[j][i][2]==2:
                if i in [2,4,6,8,10,12,14,16,18]:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, left_vis_color, thickness=-1) # left red
                elif i in [0,1]:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, middle_vis_color, thickness=-1) # black
                else:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, right_vis_color, thickness=-1) # 
            elif skeletons[j][i][2]==1:
                if i in [2,4,6,8,10,12,14,16,18]:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, left_invis_color, thickness=-1) # 
                elif i in [0,1]:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, middle_invis_color, thickness=-1) # 
                else:
                    cv.circle(canvas, tuple(skeletons[j][i, 0:2]), 4, right_invis_color, thickness=-1) # blue

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

def display_keypoint(vis_one_image, image, head_boxes, body_boxes, keypoints, labels=[], save_name=None):
    if len(body_boxes) == 0:
        return
    if len(keypoints) > 0:
        assert(len(body_boxes) == len(keypoints))
    if len(labels) > 0:
        assert(len(body_boxes) == len(labels))
    image = np.array(image)

    for idx, bbox in enumerate(head_boxes):
        colors_gt = (255,128,0) # orange
        image = cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors_gt, 2)
        #handle.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=colors_gt, linewidth=2))

    for idx, bbox in enumerate(body_boxes):
        colors_gt = (255, 255, 0)# yellow
        image = cv.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors_gt, 2)
        #handle.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor=colors_gt, linewidth=2))

    image = plot_poses(image, np.array(keypoints))

    image = Image.fromarray(image)
    if not vis_one_image:
        image.save(save_name)
    else:
        _, handle = plt.subplots(figsize=(12, 12))
        handle.imshow(image, aspect='equal')
        handle.set_title('%d body_boxes'%(len(body_boxes)), fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    RootFolder = '../RawImages'
    LabelFolder = '../Images'
    NoLabelFolder = '../NoLabelImage'
    MetaDataFolder = '../MetaData'
    VisDataFolder = '../ImagesWithBbxVis'
    batch = 'Batch_4'
    split = 'train'
    #vis_one_image = True
    vis_one_image = False

    if not vis_one_image:
        if not os.path.exists(VisDataFolder): 
            os.mkdir(VisDataFolder)
        if not os.path.exists('{}/{}'.format(VisDataFolder, batch)): 
            os.mkdir('{}/{}'.format(VisDataFolder, batch))

    kpts = keyPointVisDataset(batch,split)
    for rec in kpts.load_bbox_keypoint_vis_gt(): 
        if not vis_one_image:
            folder = rec['image_name'].split('/')[1]
            if not os.path.exists('{}/{}/{}'.format(VisDataFolder, batch, folder)): 
                os.mkdir('{}/{}/{}'.format(VisDataFolder, batch, folder))
            save_name = '{}/{}'.format(VisDataFolder, rec['image_name'])
        else:
            save_name=None
        image = kpts.extract_image(rec)
        #print(rec['image_name'])
        display_keypoint(vis_one_image, image, rec['HeadBoundingbox'], rec['BodyBoundingbox'], rec['KeyPoints'], rec['category_id'], save_name)
