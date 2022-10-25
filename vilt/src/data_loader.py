from torch.utils.data import Dataset
from transformers import ViltProcessor
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
import ast
import os
from PIL import Image
import pandas as pd

class SSTDataset(Dataset):
    def __init__(self, image_id_df, text_df, textodp_df, imageodp_df, tags_df, root_dir, ODP_T, ODP_I):

        self.image_id_df = image_id_df
        self.text_df = text_df
        self.textodp_df = textodp_df
        self.imageodp_df = imageodp_df
        self.tags_df = tags_df
        self.ODP_T = ODP_T
        self.ODP_I = ODP_I

        self.root_dir = root_dir
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        self.trans = torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)

    def __len__(self):
        return len(self.image_id_df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.image_id_df.iloc[idx]['image_id']))
        image = Image.open(img_name).convert("RGB")
        if len(image.split()) == 1:
            tf = transforms.ToTensor()
            tf_pil_img = tf(image)
            trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x)
            image = to_pil_image(trans(tf_pil_img)) # Check 이미지 형식 확인(이미지가 깨지지 않는지)
        text = str(self.text_df.iloc[idx]['text'])
        textodp = str(self.textodp_df.iloc[idx]['textodp'])
        imageodp = str(self.imageodp_df.iloc[idx]['imageodp'])
        if self.ODP_T == True:
            text = text + " <SEP> " + textodp
        if self.ODP_I == True:
            text = text + " <SEP> " + imageodp
        labels = ast.literal_eval(self.tags_df.iloc[idx]['hashtag_multihot']) # 0,1,1,0
        label = ast.literal_eval(self.tags_df.iloc[idx]['hashtag_onehot']) # 0,1,0,0 / 0,0,1,0
        # try:
        process_output = self.processor(image, text, truncation=True, padding='max_length', return_tensors="pt")
        # except:
            # print(img_name)
            # print("이미지 이름입니당 위가")

        for k, v in process_output.items():
            process_output[k] = v.squeeze() # 1,3 -> 3 벡터 쉐잎이 dim문제 배치변경하면서 확인.
        process_output['label'] = label
        process_output['labels'] = labels

        return process_output


def collate_fn(batch):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    input_ids = []
    pixel_values = []
    attention_mask = []
    token_type_ids = []
    labels = []
    label = []
    for item in batch:
        input_ids.append(item['input_ids'])
        pixel_values.append(item['pixel_values'])
        attention_mask.append(item['attention_mask'])
        token_type_ids.append(item['token_type_ids'])
        labels.append(item['labels'])
        label.append(item['label'])

    # create padded pixel values and corresponding pixel mask
    encoding = processor.feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt") # 확인 필요 (기능)

    # create new batch
    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.LongTensor(labels)
    batch['label'] = torch.LongTensor(label)

    return batch