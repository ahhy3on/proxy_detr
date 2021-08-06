# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
import json
import random
from datasets.dataset_cfg import PASCALCLASS,ID2CLASS,CLASS2ID,PASCALCLASSID,PASCALCLASS_BASEID,PASCALCLASS_NOVELID
from tqdm import tqdm
from datasets.coco import generate_episode

class EpisodicSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    only use Train
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset)))
        self.shuffle = shuffle
        self.train_mode ='base_train'
        if 'base' in self.train_mode:
            file_name = 'base_train'
            self.kshot = 150
        elif 'meta' in self.train_mode:
            file_name = 'meta_train'
            self.kshot = 10

        file_exist = os.path.isfile('./data/VOCdevkit/'+file_name+'.json')
        if not file_exist:
            print(self.train_mode+" file not exist")
            generate_episode('./data/VOCdevkit/trainset_voc.json',self.train_mode,self.kshot)
        else:
            print('already exist')
            
        self.data = json.load(open('./data/VOCdevkit/'+file_name+'.json'))

        from pycocotools.coco import COCO
        self.coco = COCO('./data/VOCdevkit/trainset_voc.json')
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.filter_no_annotation()

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        for idx in indices:
            result = self.sample_batch(idx)
            if result is None:
                continue
            else:
                yield result
    
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

    def sample_batch(self,idx):
        batch = []
        coco = self.coco
        ann_ids = coco.getAnnIds(imgIds=[idx])
        annos = coco.loadAnns(ann_ids)
        if 'base' in self.train_mode:
            annos = [anno for anno in annos if int(anno['category_id']) in PASCALCLASS_BASEID]
        if len(annos)==0:
            return None
        selected_annotation = random.choice(annos)
        positive_sample_category_id = selected_annotation['category_id']# positive support
        
        sample_idx = random.randrange(self.kshot)
        batch.append(self.data[str(positive_sample_category_id)]['annotations'][sample_idx]['id'] )

        if 'base' in self.train_mode:
            remain_id = PASCALCLASS_BASEID
        else:
            remain_id = PASCALCLASSID
        remain_id.remove(positive_sample_category_id)
        remain_id_list = random.sample(remain_id,self.batch_size-2)
        
        for remain_id_ in remain_id_list:#suuport
            batch.append(self.data[str(remain_id_)]['annotations'][sample_idx]['id'])
        
        batch.append(selected_annotation['id'])# query
        
        return batch

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
