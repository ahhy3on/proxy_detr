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
from torchvision.datasets import VOCDetection as TvVOCDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms as T
import tqdm
import random
import json
import os
import sys
import copy

PASCALCLASS = [
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "airplane", #aeroplane
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorcycle", #motorbike
    "train",
    "bottle",
    "chair",
    "dining table",
    "potted plant",
    "couch", #sofa
    "tv", #tv/monitor
]

ID2CLASS = {
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",
        10: "traffic light",
        11: "fire hydrant",
        13: "stop sign",
        14: "parking meter",
        15: "bench",
        16: "bird",
        17: "cat",
        18: "dog",
        19: "horse",
        20: "sheep",
        21: "cow",
        22: "elephant",
        23: "bear",
        24: "zebra",
        25: "giraffe",
        27: "backpack",
        28: "umbrella",
        31: "handbag",
        32: "tie",
        33: "suitcase",
        34: "frisbee",
        35: "skis",
        36: "snowboard",
        37: "sports ball",
        38: "kite",
        39: "baseball bat",
        40: "baseball glove",
        41: "skateboard",
        42: "surfboard",
        43: "tennis racket",
        44: "bottle",
        46: "wine glass",
        47: "cup",
        48: "fork",
        49: "knife",
        50: "spoon",
        51: "bowl",
        52: "banana",
        53: "apple",
        54: "sandwich",
        55: "orange",
        56: "broccoli",
        57: "carrot",
        58: "hot dog",
        59: "pizza",
        60: "donut",
        61: "cake",
        62: "chair",
        63: "couch",
        64: "potted plant",
        65: "bed",
        67: "dining table",
        70: "toilet",
        72: "tv",
        73: "laptop",
        74: "mouse",
        75: "remote",
        76: "keyboard",
        77: "cell phone",
        78: "microwave",
        79: "oven",
        80: "toaster",
        81: "sink",
        82: "refrigerator",
        84: "book",
        85: "clock",
        86: "vase",
        87: "scissors",
        88: "teddy bear",
        89: "hair drier",
        90: "toothbrush",
    }

CLASS2ID = {v: k for k, v in ID2CLASS.items()}

PASCALCLASSID = [CLASS2ID[obj] for obj in PASCALCLASS]
print(PASCALCLASSID)
#COCOWITHOUTPASCALID = [k for k,v in ID2CLASS if k not in PASCALCLASSID]

class VOCDetection(TvVOCDetection):
    def __init__(self,image_set,transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(VOCDetection, self).__init__('./data',image_set=image_set,download=True)
        self._transforms = transforms
        self.prepare = ConvertVOC(return_masks)

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

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
    def __init__(self, img_folder, ann_file, transforms, seed,return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection_Fewshot, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.ann_file=ann_file
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.seed = seed
        self.novel_seed=0
        self.batch_size=6
        self.coco_fewshot = None
        if seed>0:
            self.generate_episodic_epoch()


    def generate_episodic_epoch(self):
        random.seed(self.seed)
        kshot=5
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
                #if a['image_id'] not in temp_images_list:#중복된 이미지가 없도록
                    if a['image_id'] in img_ids:
                        img_ids[a['image_id']].append(a)
                    else:
                        img_ids[a['image_id']] = [a]

            sample_shots = []
            sample_imgs = []
            print(len(img_ids.keys()))
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

        temp_images=[]
        temp_annos=[]
        #print(selected_anno[1]['images'][-1])
        random.seed(self.seed)
        available_ID = list(ID2CLASS.keys())
        random.shuffle(available_ID)
        for category_id in available_ID:
            idx=random.randrange(kshot)
            temp_annos.append(selected_anno[category_id]['annotations'][idx])
            #temp_annos.append(selected_anno[category_id]['annotations'][idx-1])
            remain_id = list(ID2CLASS.keys())
            remain_id.remove(category_id)
            remain_id_list = random.sample(remain_id,self.batch_size-2)
            for remain_id_ in remain_id_list:
                temp_annos.append(selected_anno[remain_id_]['annotations'][idx])
            temp_annos.append(selected_anno[category_id]['annotations'][idx-1])

        for k in list(ID2CLASS.keys()):
            temp_images.extend(selected_anno[k]['images'])
        print(len(temp_images))
        print('annos ',len(temp_annos))
        #sys.exit(1)
        for i,a in enumerate(temp_annos):
            print(a['image_id'])
            if i==9:
                break
        current_set['images'] = temp_images
        current_set['annotations'] = temp_annos
        anno_path='./data/coco/'+str(self.seed)+'.json'
        
        with open(anno_path, 'w') as f:
            json.dump(current_set, f)

        self.coco_fewshot = COCO(anno_path)
        self.ids = [a['id'] for a in temp_annos]
        #random.seed(self.seed)
        #random.shuffle(self.ids)

    def __getitem__(self, idx):#batch size = 10
        #print(idx)
        if (idx%self.batch_size)==(self.batch_size-1):#query
            coco = self.coco
            ann_ids= self.ids[idx]
            target_img_id = int(coco.loadAnns(ann_ids)[0]['image_id'])
            ann_ids= coco.getAnnIds(imgIds=[target_img_id])
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs([target_img_id])[0]['file_name']
            img = self.get_image(path)
        else:#support
            coco_fewshot = self.coco_fewshot
            ann_ids= self.ids[idx]
            target = coco_fewshot.loadAnns(ann_ids)
            if target[0]['image_id']==351549:
                print('!!!!!!!!',target)
            path = coco_fewshot.loadImgs([int(target[0]['image_id'])])[0]['file_name']
            img = self.get_image(path)
            #print(img.size)
        image_id = int(target[0]['image_id'])
        target = {'image_id': image_id, 'annotations': target}
        #if idx%self.batch_size==0:#query
        #    target['query'] = True
        #else:#support
        #    target['query'] = False

        img, target = self.prepare(img, target)
        #if target['annotations'][0]['image_id']==351549:
        #        print('!!!!!!!!',target)
        #print(target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if (idx%self.batch_size)==(self.batch_size-1):#query
            target['query'] = torch.tensor(True)
        else:#support
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
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
