import os
from os.path import join, dirname
import json
import logging

import torch

from .utils import video2feats, moment_to_iou2d, embedding, video2clips
#from .video_dataset import VideoFrameDataset, ImglistToTensor

import torchvision.transforms as transforms
import time

class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size, nb_frames):
        """
        root: video root
        All arguments specified in ./clip/config/paths_catalog.py and passed by ./clip/data/__init__.py:build_dataset
        """
        super(ActivityNetDataset, self).__init__()
        with open(ann_file, 'r') as f:
            annos = json.load(f)
        
        #-------------- For debug (sample part data)----------------------
        ann_keys = tuple(annos.keys())
        sliceAnn = {}
        for idx in range(200):
            sliceAnn[ann_keys[idx]] = annos[ann_keys[idx]]
        annos = sliceAnn
        #------------------------------------------------------------------

        self.annos = []
        logger = logging.getLogger("clip.trainer")
        logger.info("Preparing data, please wait...")
        stime = time.time()
        for vid, anno in annos.items():
            duration = anno['duration']
            # Produce annotations
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                iou2d = moment_to_iou2d(moment, num_clips, duration) 
                query = embedding(sentence)
                self.annos.append(
                    {
                        'vid': vid,
                        'moment': moment,
                        'iou2d': iou2d,
                        'sentence': sentence,
                        'query': query,
                        'wordlen': query.size(0),
                        'duration': duration,
                    }
                 )
        print("Load anno time: {}".format(time.time() - stime))
        self.transforms = transforms.Compose([
            transforms.Resize([171, 128]),
            transforms.RandomCrop([112, 112]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.clips = video2clips(root, annos.keys(), self.transforms, nb_frames)
        #self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="activitynet")

    def __getitem__(self, idx):
        anno = self.annos[idx]
        vid = anno['vid']
        #return self.feats[vid], anno['query'], anno['wordlen'], anno['iou2d'], idx
        return self.clips[vid], anno['query'], anno['wordlen'], anno['iou2d'], idx
    
    def __len__(self):
        return len(self.annos)
    
    def get_duration(self, idx):
        return self.annos[idx]['duration']
    
    def get_sentence(self, idx):
        return self.annos[idx]['sentence']
    
    def get_moment(self, idx):
        return self.annos[idx]['moment']
    
    def get_vid(self, idx):
        return self.annos[idx]['vid']
