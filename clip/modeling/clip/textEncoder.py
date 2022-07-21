import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class TextEncoder(nn.Module):
    def __init__(self, model_name, pretrained, trainable):
        super().__init__()
        if pretrained:
            #self.model = DistilBertModel.from_pretrained(model_name)
            #self.model.save_pretrained('datasets/distilbert')
            self.model = DistilBertModel.from_pretrained('datasets/distilbert')
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :], last_hidden_state[:, 1:-1, :]


def build_textEncoder(cfg):
    model_name = cfg.MODEL.CLIP.TEXTEMBED.MODEL
    pretrained = cfg.MODEL.CLIP.TEXTEMBED.PRETRAIN
    trainable = cfg.MODEL.CLIP.TEXTEMBED.TRAINABLE
    return TextEncoder(model_name, pretrained, trainable)
