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
import tqdm
import random
import json
import os
import sys
import copy
from dataset_cfg import PASCALCLASS,ID2CLASS,CLASS2ID,PASCALCLASSID


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

        if train_mode =='base_train':
            self.kshot = 200
            self.seed = seed
        elif train_mode == 'base_val_code':
            self.seed = 0
            self.kshot = 10
            self.batch_size=10
        elif train_mode == 'base_val':
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
        elif train_mode == 'meta_val':
            self.seed = 0
            self.kshot = kshot
            self.batch_size=1
        
        self.train_mode = train_mode
        


    def generate_episodic_epoch(self):
        random.seed(self.seed)
        ############################
        data_path = self.ann_file
        data = json.load(open(data_path))

        new_all_cats = []
        for cat in data['categories']:
            new_all_cats.append(cat)

        id2img = {}
        for i in data['images']:
            id2img[i['id']] = i

        anno = {i: [] for i in ID2CLASS.keys()}
        for a in data['annotations']:
            if a['iscrowd'] == 1:
                continue
            anno[a['category_id']].append(a)

        selected_anno = {}

        category = ID2CLASS.keys()
        temp_images_list=[]
        for c in category:
            if c in PASCALCLASSID:
                random.seed(self.novel_seed)
            else:
                random.seed(self.seed)

            img_ids = {}
            for a in anno[c]:
                    if a['image_id'] in img_ids:
                        img_ids[a['image_id']].append(a)
                    else:
                        img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            print(len(img_ids.keys()))
            for shots in [self.kshot]:
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
                print(len(sample_shots))
                new_data = {
                    #'info': data['info'],
                    #'licenses': data['licenses'],
                    'images': sample_imgs,
                    'annotations': sample_shots,
                }
                #new_data['categories'] = new_all_cats
            temp_images_list.extend(sample_imgs)
            #print(c, new_data)
            selected_anno[c] = new_data

        current_set = {
                'info': data['info'],
                'licenses': data['licenses'],
                'categories' : new_all_cats
            }
        #############################
        self.selected_shots = selected_anno

        temp_images=[]
        temp_annos=[]

        for k in list(ID2CLASS.keys()):
            if 'base' in self.train_mode:
                if c not in PASCALCLASSID:
                    temp_images.extend(selected_anno[k]['images'])
                    temp_annos.extend(selected_anno[k]['annotations'])
            else:
                temp_images.extend(selected_anno[k]['images'])
                temp_annos.extend(selected_anno[k]['annotations'])

        current_set['images'] = temp_images
        current_set['annotations'] = temp_annos
        anno_path='./data/coco/'+self.train_mode+'.json'
        
        with open(anno_path, 'w') as f:
            json.dump(current_set, f)

        self.coco_fewshot = COCO(anno_path)
        if self.train_mode =='base_train':
            pass
        elif self.train_mode == 'base_val_code':
            self.ids = [a['id'] for a in temp_annos]

        elif self.train_mode == 'base_val':
            kshot_sample = [a['id'] for a in temp_annos]
            self.ids = [t for t in self.ids if t not in kshot_sample]

        elif self.train_mode == 'meta_train':
            self.ids = [a['id'] for a in temp_annos]

        elif self.train_mode == 'meta_val_code':
            """
            train의 k장으로 대체
            """
            self.ids = [a['id'] for a in temp_annos]
        elif self.train_mode == 'meta_val':

            kshot_sample = [a['id'] for a in temp_annos]
            self.ids = [t for t in self.ids if t not in kshot_sample]
        #self.ids = [a['id'] for a in temp_annos]


    def __getitem__(self, idx):#batch size = 10
        #print(idx)
        imgs=[]
        targets=[]
        query_category = None
        for i in range(self.batch_size):

            if 'val_code' in self.train_mode:
                coco_fewshot = self.coco_fewshot
                ann_ids= self.ids[idx]
                target = coco_fewshot.loadAnns(ann_ids)
                path = coco_fewshot.loadImgs([int(target[0]['image_id'])])[0]['file_name']
                img = self.get_image(path)

            elif 'val' in self.train_mode:
                coco = self.coco
                ann_ids= self.ids[idx]
                target_img_id = int(coco.loadAnns(ann_ids)[0]['image_id'])
                ann_ids= coco.getAnnIds(imgIds=[target_img_id])
                target = coco.loadAnns(ann_ids)
                path = coco.loadImgs([target_img_id])[0]['file_name']
                img = self.get_image(path)
                if i==1:
                    break
            else:
                if i==0:
                    coco = self.coco
                    ann_ids= self.ids[idx]
                    target_img_id = int(coco.loadAnns(ann_ids)[0]['image_id'])
                    ann_ids= coco.getAnnIds(imgIds=[target_img_id])
                    target = coco.loadAnns(ann_ids)
                    path = coco.loadImgs([target_img_id])[0]['file_name']
                    img = self.get_image(path)
                elif i==1:#positive support
                    coco_fewshot = self.coco_fewshot
                    query_category = targets[0]['labels'][random.randint(len(targets[0]['labels']))]
                    ann_ids = random.choice(self.selected_shots[query_category]['annotations'])['id']
                    target = coco_fewshot.loadAnns(ann_ids)
                    path = coco_fewshot.loadImgs([int(target[0]['image_id'])])[0]['file_name']
                    img = self.get_image(path)
                else:#random support
                    available_ID = list(ID2CLASS.keys())
                    available_ID.remove(query_category)
                    available_ID = random.sample(available_ID,self.batch_size-2)

                    coco_fewshot = self.coco_fewshot
                    ann_ids= random.choice(self.selected_shots[available_ID[i-2]]['annotations'])['id']
                    target = coco_fewshot.loadAnns(ann_ids)
                    path = coco_fewshot.loadImgs([int(target[0]['image_id'])])[0]['file_name']
                    img = self.get_image(path)

            image_id = int(target[0]['image_id'])
            target = {'image_id': image_id, 'annotations': target}

            img, target = self.prepare(img, target)

            if self._transforms is not None:
                img, target = self._transforms(img, target)
            if i==0:#query
                target['query'] = torch.tensor(True)
            else:#support
                target['query'] = torch.tensor(False)
            
            imgs.append(img)
            targets.append(target)

        imgs = imgs.reverse()
        targets = targets.reverse()
        
        imgs = torch.stack(imgs)
        targets = torch.stack(targets)

        return imgs, targets


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
        #"train": (root / "VOCdevkit/VOC2012/JPEGImages", root / 'VOCdevkit/VOC2012/JPEGImages/train.json'),
        #"val": (root / "VOCdevkit/VOC2012/JPEGImages", root / 'VOCdevkit/VOC2012/JPEGImages/val.json'),
        "train" : (root / "coco/train2017", root/"coco/annotations"/f'{mode}_train2017.json'),
        "val" : (root/ "coco/val2017", root/"coco/annotations"/f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection_Fewshot(img_folder, ann_file, transforms=make_coco_transforms(image_set), seed=seed,return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), batch_size=args.batch_size, kshot=args.kshot, train_mode =args.train_mode)
    return dataset
