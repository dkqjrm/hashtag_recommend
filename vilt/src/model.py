import torch
import torch.nn as nn
from transformers import ViltModel

class ClassificationModel(nn.Module):
    def __init__(self, pretrained_model='dandelin/vilt-b32-mlm', dropout_prob = 0.5):
        super(ClassificationModel, self).__init__()
        self.vilt = ViltModel.from_pretrained(pretrained_model)
        self.linear = nn.Linear(768, 1000)
        self.norm = nn.LayerNorm(1000)
        self.acti = nn.GELU()
        self.linear2 = nn.Linear(1000, 3896)
        self.dropout1 = nn.Dropout(dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            pixel_values=None,
            pixel_mask=None,
            head_mask=None,
            inputs_embeds=None,
            image_embeds=None,
            image_token_type_idx=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        pooler_output = self.vilt(input_ids, token_type_ids, attention_mask, pixel_values, pixel_mask).pooler_output
        predict = self.linear(pooler_output)
        predict = self.norm(predict)
        predict = self.acti(predict)
        predict = self.dropout1(predict)
        predict = self.linear2(predict)
        return predict