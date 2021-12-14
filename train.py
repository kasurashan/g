import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from argument import get_args
from backbone import vovnet57
from dataset import COCODataset, collate_fn
from model import FCOS
from transform import preset_transform
from evaluate import evaluate
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    DistributedSampler,
    all_gather,
)

###### CUDA_VISIBLE_DEVICES=1 python train.py data/coco --batch 4 --epoch 2


def accumulate_predictions(predictions):
    all_predictions = all_gather(predictions)

    if get_rank() != 0:
        return

    predictions = {}

    for p in all_predictions:
        predictions.update(p)

    ids = list(sorted(predictions.keys()))

    if len(ids) != ids[-1] + 1:
        print('Evaluation results is not contiguous')

    predictions = [predictions[i] for i in ids]

    return predictions


@torch.no_grad()
def valid(args, epoch, loader, dataset, model, device):
    if args.distributed:
        model = model.module

    print(args)

    torch.cuda.empty_cache()

    model.eval()

    pbar = tqdm(loader, dynamic_ncols=True)

    preds = {}



    for images, targets, ids in pbar:
        model.zero_grad()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        print(len(images.tensors))
        print(len(images.sizes))

        pred, _ = model(images.tensors, images.sizes)

        print(1)

        pred = [p.to('cpu') for p in pred]

        preds.update({id: p for id, p in zip(ids, pred)})

    preds = accumulate_predictions(preds)

    if get_rank() != 0:
        return

    evaluate(dataset, preds)


def train(args, epoch, loader, model, optimizer, device):
    model.train()

    if get_rank() == 0:
        pbar = tqdm(loader, dynamic_ncols=True)

    else:
        pbar = loader


    

    for images, targets, _ in pbar:
        model.zero_grad()

        images = images.to(device)

        targets2 = targets   ####### i add this for keypoint 

        #print(type(targets), targets)
        #print("train.pyy&&&&&&&&&&&&&", targets[0].key, targets[0].box ) 
        #print("train.pyy&&&&&&&&&&&&&", targets[1].key, targets[1].box )
        #print("train.pyy&&&&&&&&&&&&&", targets[2].key, targets[2].box )
        #print("train.pyy&&&&&&&&&&&&&", targets[3].key, targets[3].box )

        #print("DEVICE!!!!!", targets[0], targets[0].to(device))
        #print(targets[0].box, targets[0].key)
        #print(targets[0].to(device).box, targets[0].to(device).key)


        
        
        targets = [target.to(device) for target in targets]

            
        
        

        #print(targets)
        #print("!!!!!!train.pyy&&&&&&&&&&&&&", targets[0].key, targets[0].box )

        #print("target!!!!!!!!!", len(targets), type(targets))
        #print(targets[0].box)
        #print(targets[1].box)
        #print(targets[2].box)
        #print(targets[3].box)
        #print(targets[3].key)

        _, loss_dict = model(images.tensors, targets=targets)
        loss_cls = loss_dict['loss_cls'].mean()
        loss_box = loss_dict['loss_box'].mean()
        loss_center = loss_dict['loss_center'].mean()   #####

        loss = loss_cls + loss_box
        loss = loss_cls + loss_box + loss_center   #####
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        loss_reduced = reduce_loss_dict(loss_dict)
        loss_cls = loss_reduced['loss_cls'].mean().item()
        loss_box = loss_reduced['loss_box'].mean().item()
        loss_center = loss_reduced['loss_center'].mean().item()   #####

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'epoch: {epoch + 1}; cls: {loss_cls:.4f}; '
                    f'box: {loss_box:.4f}'
                    f'box: {loss_box:.4f}; center: {loss_center:.4f}'   #####
                )
            )


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return sampler.RandomSampler(dataset)

    else:
        return sampler.SequentialSampler(dataset)


if __name__ == '__main__':
    args = get_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    device = 'cuda'
    
    train_set = COCODataset(args.path, 'train', preset_transform(args, train=True))
    
    valid_set = COCODataset(args.path, 'val', preset_transform(args, train=False))

    backbone = vovnet57(pretrained=False)   # pretrained????? 
    model = FCOS(args, backbone)
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.l2,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[16, 22], gamma=0.1
    )

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        sampler=data_sampler(train_set, shuffle=True, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch,
        sampler=data_sampler(valid_set, shuffle=False, distributed=args.distributed),
        num_workers=2,
        collate_fn=collate_fn(args),
    )

    for epoch in range(args.epoch):
        train(args, epoch, train_loader, model, optimizer, device)  ###########
        #valid(args, epoch, valid_loader, valid_set, model, device)

        scheduler.step()

        if get_rank() == 0:
            torch.save(
                #{'model': model.module.state_dict(), 'optim': optimizer.state_dict()},  #depracted
                {'model': model.state_dict(), 'optim': optimizer.state_dict()},
                f'checkpoint/epoch-{epoch + 1}.pt',
            )

