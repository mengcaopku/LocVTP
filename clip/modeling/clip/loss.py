import torch
import torch.nn as nn
from torch.functional import F 

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ClipLoss(object):
    def __init__(self, temperature):
        self.temperature = temperature
        #self.Creduce = nn.Sequential(
        #    nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3,
        #              stride=1, padding=1),
        #    nn.ReLU()
        #)
        #self.f_cls = nn.Linear(1280, 256)
        self.reduce = nn.AdaptiveAvgPool1d(1)

    def __call__(self, video_embeddings, text_embeddings):
        # Convert video_embeddings: [batch_size, 10, 256] to [batch_size, 256]    
        #batch_size = video_embeddings.shape[0] 
        #video_embeddings = video_embeddings.permute(0, 2, 1)
        #video_embeddings = self.Creduce(video_embeddings)
        #video_embeddings = self.cls(video_embeddings.reshape(batch_size, -1))
        video_embeddings = self.reduce(video_embeddings.permute(0, 2, 1)).squeeze(2)

        logits = (text_embeddings @ video_embeddings.T) / self.temperature
        images_similarity = video_embeddings @ video_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


class ShiftLoss(object):
    def __init__(self, lossType='L1', num_classes=9):
        if lossType == 'L1':
            self.shiftloss = nn.L1Loss(reduction='mean')
        elif lossType == 'SL1': # Smooth L1
            self.shiftloss = nn.SmoothL1Loss(reduction='mean')
        elif lossType == 'MSE':
            self.shiftloss = nn.MSELoss(reduction='mean')
        elif lossType == 'CrossEntropy':
            self.shiftloss = nn.CrossEntropyLoss()

    def __call__(self, shiftPred, slens):
        return self.shiftloss(shiftPred, slens)


class FGLoss(object):
    def __init__(self):
        """the fine-grained contrastive loss"""
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        bs, numClip, _ = q.shape
        q = q.view(bs*numClip, -1)
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss
    
    def __call__(self, vidEmbed, posWord, negWord):
        # Version 1
        #_b, _numClip, _numWord = vidTextMatchScore.shape
        #vidTextMatchScore = vidTextMatchScore.view(_b*_numClip, _numWord)
        #_, maxIdx = torch.max(vidTextMatchScore, dim=1)
        #gt = F.one_hot(maxIdx, num_classes=_numWord)
        #return cross_entropy(vidTextMatchScore, gt).mean()
        # Version 2
        return self.NCE(vidEmbed, posWord, negWord) 

def build_cliploss(cfg):
    temperature = cfg.MODEL.CLIP.LOSS.TEMPERATURE 
    return ClipLoss(temperature) 

def build_shiftloss(cfg):
    losstype = cfg.MODEL.CLIP.LOSS.LOSSTYPE
    numclass = len(cfg.INPUT.SHIFTRATIO)
    return ShiftLoss(lossType=losstype, num_classes=numclass)

def build_fgloss(cfg):
    """"Build the loss for the fine-grained contrastive loss"""
    return FGLoss()