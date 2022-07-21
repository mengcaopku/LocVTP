import torch
from torch.nn.utils.rnn import pad_sequence

from clip.structures import TLGBatch
from transformers import DistilBertTokenizer

class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, is_train):
        self.text_tokenizer = "distilbert-base-uncased"
        #self.tokenizer = DistilBertTokenizer.from_pretrained(self.text_tokenizer)
        #self.tokenizer.save_pretrained('datasets/distilbert')
        self.tokenizer = DistilBertTokenizer.from_pretrained('datasets/distilbert')
        self.is_train = is_train

    def __call__(self, batch):
        captions = []
        images = []
        shiftImgs = []
        vidnames = []
        labels = []

        for x in batch:
            captions.append(x['caption']) 
            images.append(x['image'])
            vidnames.append(x['vidname'])
            labels.append(x['label'])
            if self.is_train is True:
                shiftImgs.append(x['shiftImg'])

        # Stack images
        if self.is_train is True:
            #import ipdb; ipdb.set_trace()
            clip = torch.stack(images)
            shiftClip = torch.stack(shiftImgs)
        else:
            clip = images
            shiftClip = None
        # Preprocess captions
        #text_tokenizer = "distilbert-base-uncased"
        #tokenizer = DistilBertTokenizer.from_pretrained(text_tokenizer)
        capTokens = self.tokenizer(
	        captions,
	        padding=True,
	        truncation=True,
	        max_length=512,
	        return_tensors="pt" # change to "tf" for got TensorFlow
        )
        #for key, value in batch.items():
	    #    print(f"{key}: {value.numpy().tolist()}")
        labels = torch.tensor(labels)
        return clip, shiftClip, capTokens, vidnames, labels, captions

        '''
        # -------- OLD -------------
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        clips, queries, wordlens, iou2d, idxs = transposed_batch
        return TLGBatch(
            #feats=torch.stack(clips).float(),
            clips=clips,
            queries=pad_sequence(queries).transpose(0, 1),
            wordlens=torch.tensor(wordlens),
        ), torch.stack(iou2d), idxs
        '''