import torch
from torch import nn
import torchvision
import numpy as np
import cv2
import torch.nn.functional as F

#from .featpool import build_featpool
from .vidEncoder import build_videoEncoder
from .textEncoder import build_textEncoder
from .projection import build_projection
#from .feat2d import build_feat2d
#from .integrator import build_integrator
#from .predictor import build_predictor
from .shiftpredict import build_shiftpredict
from .loss import build_cliploss, build_shiftloss, build_fgloss

def plotImg(img_path, img):
    """
    plot normalized img with shape [n, c, h, w] ranged [-1,1].
    """
    img = torchvision.utils.make_grid(img).numpy()
    img = np.transpose(img, (1,2,0))
    img += np.array([1, 1, 1])
    img *= 127.5
    img = img.astype(np.uint8)
    img = img[:, :, [2,1,0]]
    cv2.imwrite(img_path, img)

class CLIP(nn.Module):
    def __init__(self, cfg):
        super(CLIP, self).__init__()  
        self.videoEncoder = build_videoEncoder(cfg)
        self.textEncoder = build_textEncoder(cfg)
        self.vidProjector, self.textProjector = build_projection(cfg)
        self.similarityNorm = nn.Softmax(dim=2)
        #self.featpool = build_featpool(cfg)
        #self.feat2d = build_feat2d(cfg)
        #self.integrator = build_integrator(cfg)
        #self.predictor = build_predictor(cfg, self.feat2d.mask2d)
        self.shiftpredict = build_shiftpredict(cfg)
        self.cliploss = build_cliploss(cfg)
        self.shiftloss = build_shiftloss(cfg)
        self.fgloss = build_fgloss(cfg)
        self.signProjWord = nn.Linear(1, 256)
        self.distProjWord = nn.Linear(1, 256)
        self.signProjVid = nn.Linear(1, 256)
        self.distProjVid = nn.Linear(1, 256)
    '''
    def avgfeats(featlist, num_clip=256):
        avgedfeats = []
        for feat in featlist:
            feat = feat.permute(1,0).unsqueeze(0)
            avgedfeats.append(F.interpolate(feat, size=num_clip))
        return torch.stack(avgedfeats)
    '''
    def forward(self, clips, sclips, capTokens, label, captions):
        """
        Arguments:
            clips: video clip tensor feature;
            sclips: shifted video clip tensor features;
            slens: shift distance; (for regression)
            label: shift distance idx in SHIFTRATIO list; (for classicication)
            copTokens: dict with keys 'input_ids' and 'attention_mask'
        Returns:

        """
        vis_idx = 0
        #plotImg("debugVis/{}_clip0.png".format(vis_idx), clips[0].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clip1.png".format(vis_idx), clips[1].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clip2.png".format(vis_idx), clips[2].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clip3.png".format(vis_idx), clips[3].permute(1,0,2,3).cpu())
        B, C, T, H, W = clips.shape
        T = T // 16 # Number of snippet
        B_idx = torch.randperm(B).cuda()
        # Shuffle along batch-dim
        clips_delta = clips[B_idx, :]
        offset_labels = []
        clips_new = []
        for i in range(B):
            T_idx = torch.randperm(T)[0].item()
            offset_labels.append(T_idx)
            clips_retain = clips[i, :, :T_idx*16, :, :]
            clips_shiftin = clips_delta[i, :, T_idx*16:, :, :]
            clips_new_i = torch.cat((clips_retain, clips_shiftin), dim=1)
            clips_new.append(clips_new_i)
        clips_new = torch.stack(clips_new)
        offset_labels = torch.tensor(offset_labels).cuda()
        #plotImg("debugVis/{}_clipNew0_{}.png".format(vis_idx, offset_labels[0]), clips_new[0].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clipNew1_{}.png".format(vis_idx, offset_labels[1]), clips_new[1].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clipNew2_{}.png".format(vis_idx, offset_labels[2]), clips_new[2].permute(1,0,2,3).cpu())
        #plotImg("debugVis/{}_clipNew3_{}.png".format(vis_idx, offset_labels[3]), clips_new[3].permute(1,0,2,3).cpu())
        vis_idx = vis_idx + 1
        vidfeat = self.videoEncoder(clips) # [batch_size, num_clip, 4096]
        svidfeat = self.videoEncoder(clips_new) # [batch_size, num_clip, 4096]
        sentfeat, wordfeat = self.textEncoder(
            input_ids=capTokens["input_ids"], attention_mask=capTokens["attention_mask"]
        ) # [batch_size, 768], # [batch_size, num_word, 768]

        vidEmbed = self.vidProjector(vidfeat) # [batch_size, num_clip, 256]
        svidEmbed = self.vidProjector(svidfeat) # [batch_size, num_clip, 256]
        sentEmbed = self.textProjector(sentfeat) # [batch_size, 256]
        _b, _n, _ = wordfeat.shape
        wordEmbed = self.textProjector(wordfeat.reshape(_b*_n, -1)).reshape(_b, _n, -1) # [batch_size, num_word, 256]
        vidwordSim = torch.bmm(vidEmbed, wordEmbed.permute(0,2,1))
        vidwordSim = self.similarityNorm(vidwordSim)
        #---------- posWord, negWord -------
        val, idx = torch.max(vidwordSim, dim=2)

        posWord = []
        negWord = []
        multipleNeg = 4
        for _batch in range(idx.shape[0]):
            for _vid in range(idx.shape[1]):
                posWord.append(wordEmbed[_batch, idx[_batch, _vid], :])
                for _ in range(multipleNeg):
                    negWord.append(wordEmbed[_batch, np.random.choice(range(_n)), :])
        posWord = torch.stack(posWord)
        negWord = torch.stack(negWord).view(posWord.shape[0], -1, posWord.shape[1])

        #---------- Shift posWord -------
        shiftPosWord = []
        #shiftNegWord = []
        shift_idx = torch.randint(low=0, high=vidwordSim.shape[2], size=idx.shape).cuda()
        deltaWord = shift_idx - idx
        signDeltaWord = torch.sign(deltaWord).view(-1, 1).type(torch.cuda.FloatTensor)
        deltaWord = torch.abs(deltaWord).view(-1, 1).type(torch.cuda.FloatTensor)
        signDeltaWordFeat = self.signProjWord(signDeltaWord).view(idx.shape[0], idx.shape[1], -1)
        deltaWordFeat = self.distProjWord(deltaWord).view(idx.shape[0], idx.shape[1], -1)
        for _batch in range(shift_idx.shape[0]):
            for _vid in range(shift_idx.shape[1]):
                tmp = wordEmbed[_batch, shift_idx[_batch, _vid], :] \
                    + signDeltaWordFeat[_batch, _vid] \
                    + deltaWordFeat[_batch, _vid]
                shiftPosWord.append(tmp)
                #for _ in range(multipleNeg):
                #    shiftNegWord.append(wordEmbed[_batch, np.random.choice(range(_n)), :])
        shiftPosWord = torch.stack(shiftPosWord)

        #---------- posClip, negClip -------

        _, idxVid = torch.max(vidwordSim, dim=1)
        posClip = []
        negClip = []
        _, _nVid, _ = vidEmbed.shape
        for _batch in range(idxVid.shape[0]):
            for _word in range(idxVid.shape[1]):
                posClip.append(vidEmbed[_batch, idxVid[_batch, _word], :])
                for _ in range(multipleNeg):
                    negClip.append(vidEmbed[_batch, np.random.choice(range(_nVid)), :])
        posClip = torch.stack(posClip)
        negClip = torch.stack(negClip).view(posClip.shape[0], -1, posClip.shape[1])

        #---------- Shift posClip -------
        shiftPosClip = []
        shift_idxVid = torch.randint(low=0, high=vidwordSim.shape[1], size=idxVid.shape).cuda()
        deltaVid = shift_idxVid - idxVid
        signDeltaVid = torch.sign(deltaVid).view(-1, 1).type(torch.cuda.FloatTensor)
        deltaVid = torch.abs(deltaVid).view(-1, 1).type(torch.cuda.FloatTensor)
        signDeltaVidFeat = self.signProjVid(signDeltaVid).view(idxVid.shape[0], idxVid.shape[1], -1)
        deltaVidFeat = self.distProjVid(deltaVid).view(idxVid.shape[0], idxVid.shape[1], -1)
        for _batch in range(shift_idxVid.shape[0]):
            for _word in range(shift_idxVid.shape[1]):
                tmp = vidEmbed[_batch, shift_idxVid[_batch, _word], :] \
                    + signDeltaVidFeat[_batch, _vid] \
                    + deltaVidFeat[_batch, _vid]
                shiftPosClip.append(tmp)
        shiftPosClip = torch.stack(shiftPosClip)

        shiftPred = self.shiftpredict(svidEmbed, sentEmbed)  # [batch_size, 10]

        if self.training:
            return self.cliploss(vidEmbed, sentEmbed), self.shiftloss(shiftPred, offset_labels), \
                self.fgloss(vidEmbed, posWord, negWord), self.fgloss(wordEmbed, posClip, negClip), \
                self.fgloss(vidEmbed, shiftPosWord, negWord), self.fgloss(wordEmbed, shiftPosClip, negClip)
        return vidEmbed, sentEmbed

        '''
        # video clip feature list to fixed size feature tensor
        avgedfeats = []
        for feat in featlist:
            feat = feat.permute(1,0).unsqueeze(0)
            avgedfeats.append(F.interpolate(feat, size=256).squeeze())
        feats = torch.stack(avgedfeats)
        feats = self.featpool(feats)
        map2d = self.feat2d(feats)
        map2d = self.integrator(batches.queries, batches.wordlens, map2d)
        scores2d = self.predictor(map2d)
        #print(self.training) 
        if self.training:
            return self.cliploss(scores2d, ious2d)
        return scores2d.sigmoid_() * self.feat2d.mask2d
        '''
