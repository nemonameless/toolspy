from mmdet.core import AnchorGenerator, anchor_target
from mmdet.core.bbox import PseudoSampler
import numpy as np
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv
import os
from mmcv import Config
from IPython import embed
from mmdet.datasets import build_dataset
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmdet.models import build_detector
import torch
import torch.nn as nn
import cv2

def bbox_overlaps(boxes1, boxes2):

    boxes1 = torch.from_numpy(boxes1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # boxes1.area()
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # boxes2.area()
    # boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class AssignResult(object):
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

class AnchorHead(nn.Module):
    def __init__(self,
                 anchor_scales=[8],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[8, 16, 32, 64, 128], # [4, 8, 16, 32, 64] in Faster RCNN
                 anchor_base_sizes=None,
                 octave_base_scale=4, #
                 scales_per_octave=3, #
                 pos_iou_low = 0.5,
                 pos_iou_high = 0.8):
        super(AnchorHead, self).__init__()
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides

        self.pos_iou_low = pos_iou_low
        self.pos_iou_high = pos_iou_high

        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.octave_scales = np.array([2**(i / scales_per_octave) for i in range(scales_per_octave)])
        self.anchor_scales = self.octave_scales * octave_base_scale # 3 scales

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales) # 3*3
        self.cls_out_channels = 80 

    def get_anchors(self, featmap_sizes, img_metas, device='cpu'):

        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.
        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        self.pos_iou_thr=0.5
        self.neg_iou_thr=0.4
        self.min_pos_iou=0
        self.gt_max_assign_all=True

        if overlaps.numel() == 0:
            raise ValueError('No gt or proposals')

        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),-1,dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)


        # 2. assign negative: below
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds_high = max_overlaps < self.pos_iou_high
        pos_inds_low = max_overlaps > self.pos_iou_low
        pos_inds = pos_inds_high*pos_inds_low

        #pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                gt_labels = torch.from_numpy(gt_labels).long()
                assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_pos_anchors(self, fpn_feats, gt_bboxes, gt_labels, img_metas, cfg, gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in fpn_feats]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = fpn_feats[0].device
        img_metas = [img_metas]

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        for i in range(len(img_metas)):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        anchor_valid = anchor_list[0][valid_flag_list[0],:] 

        overlaps = bbox_overlaps(gt_bboxes, anchor_valid)
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)

        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchor_valid, gt_bboxes)

        pos_inds = sampling_result.pos_inds
        #labels = anchor_valid.new_zeros(anchor_valid.shape[0], dtype=torch.long)

        labels_pos = gt_labels[sampling_result.pos_assigned_gt_inds]
        anchor_pos = anchor_valid[pos_inds]

        return anchor_pos, labels_pos


def get_img_meta(model, idx=2):
	cfg = model.cfg

	# get gt_ann
	dataset_train = build_dataset(cfg.data.train)
	ann_info = dataset_train.get_ann_info(idx)
	img_path = '{}{}'.format(cfg.data.train.img_prefix, ann_info['seg_map'].replace('png','jpg'))
	gt_bboxes = ann_info['bboxes']
	gt_labels = ann_info['labels']
	gt_segms = ann_info['masks']

	# get img_meta
	test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
	test_pipeline = Compose(test_pipeline)
	data = dict(img=img_path)
	data = test_pipeline(data) 
	# dict_keys(['img_meta', 'img'])
	img_metas = data['img_meta'][0].data
	# dict_keys(['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg'])
	img_tensor = data['img'][0].data

	# get FPN feat_size
	c,h,w = img_tensor.shape
	input_img_feat = img_tensor.view(1,c,h,w)
	device = input_img_feat.device
	fpn_feats = model.extract_feat(input_img_feat)

	return img_path, fpn_feats, gt_bboxes, gt_labels, img_metas, gt_segms


if __name__ == '__main__':

	config_file = './retinanet_r50_fpn_1x.py'
	checkpoint_file = None #'./work_coco/retinanet_r50_fpn_1x/epoch_12.pth'
	model = init_detector(config_file, checkpoint_file, device='cpu')

	anchor_head=AnchorHead(anchor_scales=[8],
					anchor_ratios=[0.5, 1.0, 2.0],
					anchor_strides=[8, 16, 32, 64, 128],
					anchor_base_sizes=None,
					octave_base_scale=4, #
					scales_per_octave=3, #
					pos_iou_low = 0.5,
					pos_iou_high = 0.8)

	idx = 2 # rand select
	img_path, fpn_feats, gt_bboxes, gt_labels, img_metas, gt_segms = get_img_meta(model, idx)
	anchor_pos, labels_pos = anchor_head.get_pos_anchors(fpn_feats, gt_bboxes, gt_labels, img_metas, model.cfg)


	allcolors=[[255,0,255],[255,0,0],[0,255,255],[0,255,0],[255,255,0],[0,0,255]]

	img = cv2.imread(img_path)
	width = img.shape[1]
	height = img.shape[0]
	for i,bbox in enumerate(gt_bboxes):
		x,y,w,h = bbox[:]
		x1 = int(round(float(x)))
		y1 = int(round(float(y)))
		x2 = int(round(float(w)))
		y2 = int(round(float(h)))

		cx = (x1+x2)//2
		cy = (y1+y2)//2

		textcolor = (0,0,0)#allcolors[int(gt_labels[i])%6]
		#textinfo = '{} {}'.format(int(gt_labels[i]), y2-y1)
		textinfo = '{}'.format(int(gt_labels[i]))
		img = cv2.putText(img, textinfo, (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textcolor, 3)
		img = cv2.rectangle(img, (x1,y1), (x2,y2), textcolor, 3)
		img = cv2.circle(img, (cx,cy), 1, textcolor, 3)

	for i,bbox in enumerate(anchor_pos):
		x,y,w,h = bbox[:]
		x1 = int(round(float(x)))
		y1 = int(round(float(y)))
		x2 = int(round(float(w)))
		y2 = int(round(float(h)))

		cx = (x1+x2)//2
		cy = (y1+y2)//2

		textcolor = allcolors[int(labels_pos[i]) % 6]
		#textinfo = '{} {}'.format(int(gt_labels[i]), y2-y1)
		textinfo = '{}'.format(int(labels_pos[i]))
		img = cv2.putText(img, textinfo, (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, textcolor, 2)
		img = cv2.rectangle(img, (x1,y1), (x2,y2), textcolor, 2)
		img = cv2.circle(img, (cx,cy), 1, textcolor, 2)

	cv2.imwrite('{}'.format(img_path.split('/')[-1]),img)