# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
from tqdm import tqdm
import random
import json
import os
import sys
import copy
from datasets.dataset_cfg import PASCALCLASS,ID2CLASS,CLASS2ID,PASCALCLASSID,PASCALCLASS_BASEID,PASCALCLASS_NOVELID


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        #print('target',self.ids[idx])
        image_id = int(self.ids[idx])
        #self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class CocoDetection_Fewshot(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, seed, return_masks, cache_mode=False, local_rank=0, local_size=1, batch_size=1,kshot=1,train_mode=None):
        super(CocoDetection_Fewshot, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.ann_file=ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.novel_seed=0
        self.batch_size=batch_size
        self.coco_fewshot = None
        self.selected_shots = None
        self.train_mode = train_mode
        self.filter_no_annotation()

        if train_mode =='base_train':
            self.kshot = 150
            self.seed = seed

        elif train_mode == 'base_val_code':
            self.seed = 0
            self.kshot = 10
            self.batch_size=10
            self.set_validation_support_set()

        elif train_mode == 'base_val_query':
            self.seed = 0
            self.kshot = 10
            self.batch_size=1
        elif train_mode == 'meta_train':
            self.seed = seed
            self.kshot = kshot

        elif train_mode == 'meta_val_code':
            self.seed = 0
            self.kshot = kshot
            self.batch_size = kshot
            self.set_validation_support_set()

        elif train_mode == 'meta_val_query':
            self.seed = 0
            self.kshot = kshot
            self.batch_size=1
        
    def filter_no_annotation(self):
        image_list=[]
        print("filter no annotation image like http://cocodataset.org/#explore?id=25593")
        for j in tqdm(self.ids):
            coco = self.coco
            img_id = j
            ann_ids = coco.getAnnIds(imgIds=[img_id])

            annos = coco.loadAnns(ann_ids)
            if len(annos)==0:
                continue
            image_list.append(j)
        self.ids = image_list

    def set_validation_support_set(self):
        #for caculation category code in validation

        if 'base' in self.train_mode:
            file_name = 'base_train'
        elif 'meta' in self.train_mode:
            file_name = 'meta_train'

        file_exist = os.path.isfile('./data/VOCdevkit/'+file_name+'.json')
        if not file_exist:
            generate_episode(self.ann_file,self.train_mode,self.kshot)
        data = json.load(open('./data/VOCdevkit/'+file_name+'.json'))

        available_ID = PASCALCLASSID

        if 'base' in self.train_mode:
            available_ID = [k for k in available_ID if k in PASCALCLASS_BASEID]

        annotation_list=[]
        for j in available_ID:
            data[str(j)]['annotations'] = data[str(j)]['annotations'][:self.kshot]
            data[str(j)]['images'] = data[str(j)]['images'][:self.kshot]
            annotation_list.extend([d['id'] for d in data[str(j)]['annotations']])
        
        self.ids = annotation_list

    def __getitem__(self, idx):

        coco = self.coco
        ann_ids= self.ids[idx]

        if 'train' in self.train_mode:
            if (idx%self.batch_size)==(self.batch_size-1): #query
                target_img_id = int(coco.loadAnns(ann_ids)[0]['image_id'])
                ann_ids= coco.getAnnIds(imgIds=[target_img_id])
                target = coco.loadAnns(ann_ids)
                path = coco.loadImgs([target_img_id])[0]['file_name']
                img = self.get_image(path)
            else:
                target = coco.loadAnns(ann_ids) #support
                path = coco.loadImgs([int(target[0]['image_id'])])[0]['file_name']
                img = self.get_image(path)

        elif 'val_query' in self.train_mode:
            target_img_id = ann_ids
            ann_ids= coco.getAnnIds(imgIds=[target_img_id])
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs([target_img_id])[0]['file_name']
            img = self.get_image(path)

        elif 'val_code' in self.train_mode:
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs([int(target[0]['image_id'])])[0]['file_name']
            img = self.get_image(path)

        image_id = int(target[0]['image_id'])
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        if 'train' in self.train_mode:
            if (idx%self.batch_size)==(self.batch_size-1): #query
                target['query'] = torch.tensor(True)
            else:
                target['query'] = torch.tensor(False)
        elif 'val_query' in self.train_mode:
            target['query'] = torch.tensor(True)
        elif 'val_code' in self.train_mode:
            target['query'] = torch.tensor(False)

        return img, target

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def generate_episode(data_path,train_mode,kshot):
        ############################
        data = json.load(open(data_path))

        new_all_cats = []
        for cat in data['categories']:
            new_all_cats.append(cat)

        id2img = {}
        for i in data['images']:
            id2img[i['id']] = i

        anno = {i: [] for i in PASCALCLASSID}
        for a in data['annotations']:
            if a['iscrowd'] == 1:
                continue
            anno[a['category_id']].append(a)

        k_shot_annotation_list = {}

        category = PASCALCLASS

        for c in tqdm(category):
            if c in PASCALCLASS_BASEID:
                if 'base' in train_mode:
                    continue

            img_ids = {}
            for a in anno[c]:
                if a['image_id'] in img_ids:
                    img_ids[a['image_id']].append(a)
                else:
                    img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []

            for shots in [kshot]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots)
                    
                    #print(len(imgs))
                    for img in imgs:
                        skip = False
                        for s in sample_shots:
                            if img == s['image_id']:
                                skip = True
                                break
                        if skip:
                            continue
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue
                        #print(len(id2img[img]))
                        sample_shots.extend(img_ids[img])
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break

                new_data = {
                    #'info': data['info'],
                    #'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }

                #new_data['categories'] = new_all_cats
            #print(c, new_data)
            k_shot_annotation_list[c] = new_data

        k_shot_annotation_file='./data/VOCdevkit/'+train_mode+'.json'

        with open(k_shot_annotation_file, 'w') as f:
            json.dump(k_shot_annotation_list, f)


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        #print('image id', image_id)
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        #print("bbox : ",boxes)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #[400,500,600]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),

            #T.RandomSelect(
            T.RandomResize(scales, max_size=1333),
                #T.Compose([
                #    T.RandomResize([400, 500, 600]),
                #    T.RandomSizeCrop(384, 600),
                #    T.RandomResize(scales, max_size=1333),
                #])
            #),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set,seed, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
            #"train" : (root / "cmes_new/images", root / "train_box.json"),
            #"val" : (root/"cmes_new/images", root / "val_box.json"),
        "train": (root / "VOCdevkit/VOC2012", root / 'VOCdevkit/trainset_voc.json'),
        "val": (root / "VOCdevkit/VOC2012", root / 'VOCdevkit/valset_voc.json'),
        #"train" : (root / "coco/train2017", root/"coco/annotations"/f'{mode}_train2017.json'),
        #"val" : (root/ "coco/val2017", root/"coco/annotations"/f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection_Fewshot(img_folder, ann_file, transforms=make_coco_transforms(image_set), seed=seed,return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), batch_size=args.batch_size, kshot=args.kshot, train_mode =args.train_mode)
    return dataset
