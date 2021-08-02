# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import tqdm

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
import wandb
from datasets import build_dataset, get_coco_api_from_dataset
import datasets.samplers as samplers
from torch.utils.data import DataLoader

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()


    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for idx, _ in enumerate(metric_logger.log_every(range(len(data_loader)), print_freq, header)):

        outputs = model(samples,targets)
        #print(targets)
        for j in range(len(targets)-1):
            temp = targets[j]["labels"][0]
            targets[j]["size"]=targets[-1]["size"]
            targets[j]["orig_size"]=targets[-1]["orig_size"]
            targets[j]["iscrowd"]= targets[-1]["iscrowd"][targets[-1]["labels"]==temp]
            targets[j]["area"]=  targets[-1]["area"][targets[-1]["labels"]==temp]
            targets[j]["image_id"]= targets[-1]["image_id"]
            targets[j]["boxes"] = targets[-1]["boxes"][targets[-1]["labels"]==temp]
            targets[j]["labels"] = targets[-1]["labels"][targets[-1]["labels"]==temp]
        
        targets = targets[:-1]
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        #reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        #loss_value = outputs['loss_classifier']+outputs['loss_box_reg']+outputs['loss_objectness']+outputs['loss_rpn_box_reg']
        loss_value = losses_reduced_scaled.item()

        if args.wandb:
            wandb.log({"losses" : loss_value})

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)
        samples, targets = prefetcher.next()
        #break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir,args=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    #print(iou_types)
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    #coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    #idx=0
    args.train_mode = 'base_val_code'
    dataset_val = build_dataset(image_set='val',seed=args.seed, args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_category_code = DataLoader(dataset_val, 1, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    
    category_code=[]
    category_code_label = []

    for samples, targets in tqdm(data_loader_category_code):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        category_code_mean = model(samples,targets,extract_category_code_phase=True)
        category_code.append(category_code_mean)
        temp = targets[0]["labels"][0]
        category_code_label.append(temp)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        new_targets=[]
        temp_category_code=[]

        for label in torch.unique(targets[0]["labels"]):
            
            temp_category_code.append(category_code[category_code_label.index(label)])
            temp = category_code_label.index(label)
            temp_target={}
            temp_target["size"]=targets[0]["size"]
            temp_target["orig_size"]=targets[0]["orig_size"]
            temp_target["iscrowd"]= targets[0]["iscrowd"][targets[0]["labels"]==temp]
            temp_target["area"]=  targets[0]["area"][targets[0]["labels"]==temp]
            temp_target["image_id"]= targets[0]["image_id"]
            temp_target["boxes"] = targets[0]["boxes"][targets[0]["labels"]==temp]
            temp_target["labels"] = targets[0]["labels"][targets[0]["labels"]==temp]
            new_targets.append(temp_target)
        
        targets = new_targets
        temp_category_code = torch.stack(temp_category_code)

        outputs = model(samples,targets,predict_category=temp_category_code)
        
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
