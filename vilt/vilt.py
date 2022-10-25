import torch
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import ViltProcessor, ViltForImagesAndTextClassification, ViltConfig, ViltModel, AdamW
import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import ast
import shutil
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import autocast
import torch.nn.functional as F
import argparse
import json
import logging
from torch.optim.lr_scheduler import OneCycleLR
from src.scoring import scoring
from src.model import ClassificationModel
from src.data_loader import SSTDataset, collate_fn
import pickle
import requests

#
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('image_id_csv')
parser.add_argument('text_csv')
parser.add_argument('textodp_csv')
parser.add_argument('imageodp_csv')
parser.add_argument('tags_csv')
parser.add_argument('data_image') # 이미지 폴더 위치
parser.add_argument('train_pkl') # trian 인덱스
parser.add_argument('val_pkl') # validation 인덱스
parser.add_argument('test_pkl') # test 인덱스
parser.add_argument('save_path')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--lr', type=float, default=0.0001) # 0.001, 0.0005, 0.00005, 0.00001
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--threshold', type=int, default=1)
parser.add_argument('--dropout_prob', type=float, default=0)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--resume-from')
parser.add_argument('--warmup', type=float, default=0.1)
parser.add_argument('--ODP_T', action='store_true')
parser.add_argument('--ODP_I', action='store_true')


args = parser.parse_args()

np.random.seed(42)
torch.random.manual_seed(42)

# device = torch.device('cuda')
device = torch.device(args.device)
print(device)

model = ClassificationModel()
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = model.to(device)

image_id_df = pd.read_csv(args.image_id_csv)
text_df = pd.read_csv(args.text_csv)
textodp_df = pd.read_csv(args.textodp_csv)
imageodp_df = pd.read_csv(args.imageodp_csv)
tags_df = pd.read_csv(args.tags_csv)

with open(args.train_pkl, "rb") as f:
    train_list = pickle.load(f)
train_list = list(map(int, train_list))
with open(args.val_pkl, "rb") as f:
    val_list = pickle.load(f)
val_list = list(map(int, val_list))
with open(args.test_pkl, "rb") as f:
    test_list = pickle.load(f)
test_list = list(map(int, test_list))

#df_1['post_id'].isin([1,2,3])
batch_size = args.batch_size
train_dataset = SSTDataset(image_id_df[image_id_df['post_id'].isin(train_list)], text_df[text_df['post_id'].isin(train_list)], textodp_df[textodp_df['post_id'].isin(train_list)], imageodp_df[imageodp_df['post_id'].isin(train_list)], tags_df[tags_df['post_id'].isin(train_list)], args.data_image, args.ODP_T, args.ODP_I) # '/data/user17/data/MaCon/InstagramImage'
val_dataset = SSTDataset(image_id_df[image_id_df['post_id'].isin(val_list)], text_df[text_df['post_id'].isin(val_list)], textodp_df[textodp_df['post_id'].isin(val_list)], imageodp_df[imageodp_df['post_id'].isin(val_list)], tags_df[tags_df['post_id'].isin(val_list)], args.data_image, args.ODP_T, args.ODP_I) # '/data/user17/data/MaCon/InstagramImage'
test_dataset = SSTDataset(image_id_df[image_id_df['post_id'].isin(test_list)], text_df[text_df['post_id'].isin(test_list)], textodp_df[textodp_df['post_id'].isin(test_list)], imageodp_df[imageodp_df['post_id'].isin(test_list)], tags_df[tags_df['post_id'].isin(test_list)], args.data_image, args.ODP_T, args.ODP_I) # '/data/user17/data/MaCon/InstagramImage'

train_loader = DataLoader(train_dataset, batch_size = batch_size,collate_fn=collate_fn, shuffle = True, drop_last = True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size = batch_size, collate_fn=collate_fn, shuffle = True, drop_last = True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size = batch_size,collate_fn=collate_fn, shuffle = True, drop_last = True, num_workers=16)

def save_checkpoint(state, is_best, model_save_path, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_path, 'model_best.pth.tar'))

save_path = args.save_path# '/data/user17/data/HashtagRecommend/saved_model/'
save_path = os.path.join(save_path,
                             time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))+'lr'+str(args.lr))
writer = SummaryWriter(log_dir=save_path)
with open(os.path.join(save_path, 'argparse_json'), 'w') as f:
    json.dump(args.__dict__, f)

criterion = torch.nn.CrossEntropyLoss()

# Hyperparameter
lr = args.lr
weight_decay = args.weight_decay
optimizer = AdamW(model.parameters(), lr=lr, weight_decay = weight_decay) # 대충 옵티마이저 정해야함.
scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=args.epoch, anneal_strategy='linear', pct_start=args.warmup)

threshold = args.threshold
topk = args.topk

global_steps = 0
epoch = 0
max_f1 = 0
stop_cnt = 0

while True:
    epoch += 1
    model.train()
    precision = []
    recall = []
    f1 = []
    train_loss = 0
    cnt = 0
    steps = 0
    for batch in tqdm(train_loader, total=len(train_loader)):
        # print('batch out')
        with autocast():

            batch = {k: v.to(device) for k, v in batch.items()}
            # if epoch == 1:
            #     batch['pixel_mask'] = torch.zeros_like(batch['pixel_mask']).to(device)
            logits = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['pixel_values'],
                           batch['pixel_mask'])
            label = batch['label'].to(device)
            labels = batch['labels'].to(device)
            # print(logits)
            loss = criterion(logits, label.float())
            train_loss += loss.item()

            cnt += len(label)
            model.zero_grad()
            loss.backward()
        optimizer.step()
        scheduler.step()
        writer.add_scalar(tag='learning_rate',
                          scalar_value=optimizer.param_groups[0]['lr'],
                          global_step=global_steps)
        if (steps % int((len(train_loader) * 0.5))) == 0:
            batch_p, batch_r, batch_f1 = scoring(logits, labels, topk)  # topk
            precision.extend(batch_p)
            recall.extend(batch_r)
            f1.extend(batch_f1)

            writer.add_scalar(tag='batch/batch_precision',
                              scalar_value=sum(batch_p) / len(batch_p),
                              global_step=global_steps)
            writer.add_scalar(tag='batch/batch_recall',
                              scalar_value=sum(batch_r) / len(batch_r),
                              global_step=global_steps)
            writer.add_scalar(tag='batch/batch_f1',
                              scalar_value=2 * sum(batch_p) / len(batch_p) * sum(batch_r) / len(batch_r) / (sum(batch_p) / len(batch_p) + sum(batch_r) / len(batch_r)),
                              global_step=global_steps)
            writer.add_scalar(tag='batch/batch_loss',
                              scalar_value=loss.item(),
                              global_step=global_steps)
        steps += 1
        global_steps += 1
        # print('iter end')
    writer.add_scalar(tag='train/train_learning_rate',
                      scalar_value=optimizer.param_groups[0]['lr'],
                      global_step=epoch)
    writer.add_scalar(tag='train/train_precision',
                      scalar_value=sum(precision) / len(precision),
                      global_step=epoch)
    writer.add_scalar(tag='train/train_recall',
                      scalar_value=sum(recall) / len(recall),
                      global_step=epoch)
    writer.add_scalar(tag='train/train_f1',
                      scalar_value=2 * sum(precision) / len(precision) * sum(recall) / len(recall) / (sum(precision) / len(precision) + sum(recall) / len(recall)),
                      global_step=epoch)
    writer.add_scalar(tag='train/train_loss',
                      scalar_value=train_loss / cnt,
                      global_step=epoch)

    model.eval()
    precision = []
    recall = []
    f1 = []
    val_loss = 0
    cnt = 0
    steps = 0
    for batch in tqdm(val_loader, total=len(val_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        label = batch['label'].to(device)
        labels = batch['labels'].to(device)
        # if epoch == 1:
        #     batch['pixel_mask'] = torch.zeros_like(batch['pixel_mask']).to(device)
        with torch.no_grad():
            logits = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['pixel_values'],
                           batch['pixel_mask'])
        # print(logits)
        loss = criterion(logits, label.float())
        val_loss += loss.item()
        cnt += len(label)
        steps += 1

        if (steps % int((len(val_loader) * 0.5))) == 0: # 메트릭 확인을 러닝하면서 확인..
            batch_p, batch_r, batch_f1 = scoring(logits, labels, topk)  # topk
            precision.extend(batch_p)
            recall.extend(batch_r)
            f1.extend(batch_f1)
    val_p = sum(precision) / len(precision)
    val_r = sum(recall) / len(recall)
    val_f1 = 2 * (val_p * val_r) / (val_p + val_r)
    writer.add_scalar(tag='val/val_precision',
                      scalar_value=val_p,
                      global_step=epoch)
    writer.add_scalar(tag='val/val_recall',
                      scalar_value=val_r,
                      global_step=epoch)
    writer.add_scalar(tag='val/val_f1',
                      scalar_value=val_f1,
                      global_step=epoch)
    writer.add_scalar(tag='val/val_loss',
                      scalar_value=val_loss / cnt,
                      global_step=epoch)

    if val_f1 > max_f1:
        max_f1 = val_f1
        stop_cnt = 0
        is_best = True
    else:
        stop_cnt += 1
        is_best = False

    save_checkpoint({
        'epoch': epoch,
        'model': model,
        'state_dict': model.state_dict(),
        'precision': val_p,
        'recall': val_r,
        'f1-score': val_f1,
        'optimizer': optimizer.state_dict()
    }, is_best, save_path, os.path.join(save_path, 'epoch' + str(epoch) + '.pth.tar'))
    if epoch == args.epoch or stop_cnt > threshold:
        print("Training finished.")
        print("Loading Model....")
        model_data = torch.load(os.path.join(save_path, 'model_best.pth.tar'))
        model.load_state_dict(model_data['state_dict'])
        print("Model is loaded..")
        model.eval()
        precision = []
        recall = []
        f1 = []
        test_loss = 0
        cnt = 0
        steps = 0
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch['label'].to(device)
            labels = batch['labels'].to(device)
            with torch.no_grad():
                logits = model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'],
                               batch['pixel_values'],
                               batch['pixel_mask'])
            # print(logits)
            loss = criterion(logits, label.float())
            test_loss += loss.item()
            cnt += len(label)
            steps += 1
            batch_p, batch_r, batch_f1 = scoring(logits, labels, topk)  # topk
            precision.extend(batch_p)
            recall.extend(batch_r)
            f1.extend(batch_f1)
        test_p = sum(precision) / len(precision)
        test_r = sum(recall) / len(recall)
        test_f1 = 2 * (test_p * test_r) / (test_p + test_r)
        print(test_p)
        print(test_r)
        print(test_f1)
        writer.add_scalar(tag='test/test_precision',
                          scalar_value=test_p,
                          global_step=1)
        writer.add_scalar(tag='test/test_recall',
                          scalar_value=test_r,
                          global_step=1)
        writer.add_scalar(tag='test/test_f1',
                          scalar_value=test_f1,
                          global_step=1)
        writer.add_scalar(tag='test/test_loss',
                          scalar_value=test_loss / cnt,
                          global_step=1)

        writer.flush()

        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=Code Finished')
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=lr'+str(args.lr))
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=epoch'+str(epoch))
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=test_p'+str(test_p))
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=test_recall'+str(test_r))
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=test_f1'+str(test_f1))
        requests.get(
            'https://api.telegram.org/bot5551522291:AAHc82W8WZBvUllv_IifYs9Alqwoa0GlKic/sendMessage?chat_id=5268437701&text=----------------------------')
        break
