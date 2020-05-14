import os
import sys
import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from matplotlib import pyplot as plt
from IPython import embed
dataset_dir = "/data/Dataset/coco"
subset = "train"
year = "2017"

coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
catIds = coco.getCatIds(catNms=['dog'])
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds=[1688]) # 000000001688.jpg
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

#old_img = cv2.imread("{}/{}{}/{}".format(dataset_dir, subset, year, img['file_name']))
old_img = Image.open("{}/{}{}/{}".format(dataset_dir, subset, year, img['file_name']))
old_img = np.array(old_img)
plt.imshow(old_img)
plt.axis('off')

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)

for ann in anns:
    m = coco.annToMask(ann)
    print(ann['segmentation'])

coco.showAnns(anns)
plt.savefig("./{}".format(img['file_name']))
plt.show()