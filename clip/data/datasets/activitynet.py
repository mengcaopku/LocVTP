import os
from os.path import join, dirname
import json
import logging
import numpy as np
from PIL import Image
import cv2
import albumentations as A

import torch

from .utils import video2feats, moment_to_iou2d, embedding, video2clips
from .video_dataset import VideoFrameDataset, ImglistToTensor

import torchvision.transforms as transforms
import time
import io
import tarfile
import random
from torchvision.io import read_video
'''
class ActivityNetDataset(torch.utils.data.Dataset):

    def __init__(self, ann_file, root, feat_file, num_pre_clips, num_clips, pre_query_size, nb_frames, sample_interval):
        """
        root: video root
        All arguments specified in ./clip/config/paths_catalog.py
        Using TSN video sampling way.
        """
        super(ActivityNetDataset, self).__init__()
        with open(ann_file, 'r') as f:
            annos = json.load(f)
        
        #-------------- For debug (sample part data)----------------------
        #ann_keys = tuple(annos.keys())
        #sliceAnn = {}
        #for idx in range(50):
        #    sliceAnn[ann_keys[idx]] = annos[ann_keys[idx]]
        #annos = sliceAnn
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
        self.clips = video2clips(root, annos.keys(), self.transforms, nb_frames, sample_interval)
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
'''


class VideoRecord(object):
    """
    Helper class for class ActivityNetDataset. This class
    represents a video sample's metadata.
    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, row, root_datapath):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])

    @property
    def vidname(self):
        return self._data[0]

    @property
    def path(self):
        return self._path

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame + 1  # +1 because end frame is inclusive
    @property
    def start_frame(self):
        return int(self._data[1])

    @property
    def end_frame(self):
        return int(self._data[2])

    @property
    def caption(self):
        return self._data[3]
    
    @property
    def max_frame(self):
        # the number of all frames within the video
        return int(self._data[4])

class ActivityNetDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.
    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.
    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.
    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.
    Args:
        root: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        ann_file: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.
    """
    def __init__(self,
                 root: str,
                 ann_file: str,
                 #tokenizer,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 shiftRatio: float = 0.5,
                 imagefile_template: str='{:05d}.png',
                 transform = None,
                 random_shift: bool = True,
                 test_mode: bool = False):
        super(ActivityNetDataset, self).__init__()
        """
        All arguments specified in ./clip/config/paths_catalog.py and passed by ./clip/data/__init__.py:build_dataset
        """
        self.root = root
        self.ann_file = ann_file
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = get_transforms()
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.shiftRatio = shiftRatio
        #self.tokenizer = tokenizer

        self._parse_list()

    def _load_image(self, directory, idx, tarInfo):
        ## version1: frame dataset
        #return [Image.open(os.path.join(directory, self.imagefile_template.format(idx))).convert('RGB')]

        ## version2: tar dataset (with arcname)
        #tar = tarfile.open(directory+'.tar')
        #return [Image.open(io.BytesIO(tar.extractfile(self.imagefile_template.format(idx)).read())).convert('RGB')]

        ## version3: tar dataset (without arcname)
        #tar = tarfile.open(directory+'.tar')
        #_tarfilePath = '/'.join(tar.getnames()[0].split('/')[:-1]) + '/' + self.imagefile_template.format(idx)
        #return  [Image.open(io.BytesIO(tar.extractfile(_tarfilePath).read())).convert('RGB')]

        ## version4: tar dataset (without arcname) and tarInfo pre-extracted
        _tarfilePath = '/'.join(tarInfo.getnames()[0].split('/')[:-1]) + '/' + self.imagefile_template.format(idx)
        return  [Image.open(io.BytesIO(tarInfo.extractfile(_tarfilePath).read())).convert('RGB')]
        

    def _parse_list(self):
        #self.video_list = [VideoRecord(x.strip().split(), self.root) for x in open(self.ann_file)] # Original
        video_list = []
        for x in open(self.ann_file):
            vid, sfrm, efrm, allfrm = x.strip().split('#')[0].split()
            #print(f"{vid}/{sfrm}/{efrm}")
            caption = x.strip().split('#')[1]
            video_list.append(VideoRecord([vid, sfrm, efrm, caption, allfrm], self.root))
        self.video_list = video_list

    def _sample_indices(self, record):
        """
        For each segment, chooses an index from where frames
        are to be loaded from.
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """
        segment_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if segment_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), segment_duration) + np.random.randint(segment_duration, size=self.num_segments)

        # edge cases for when a video has approximately less than (num_frames*frames_per_segment) frames.
        # random sampling in that case, which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_val_indices(self, record):
        """
        For each segment, finds the center frame index.
        Args:
            record: VideoRecord denoting a video sample.
        Returns:
             List of indices of segment center frames.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_test_indices(self, record):
        """
        For each segment, finds the center frame index.
        Args:
            record: VideoRecord denoting a video sample
        Returns:
            List of indices of segment center frames.
        """
        # Original Codes
        #tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        #offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        # Current Codes (Dense Sampling)
        offsets = np.arange(0, record.num_frames, self.frames_per_segment)

        return offsets

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.
        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """
        record = self.video_list[index]
        # new recordShift
        # Version 1: random (supervised by regression)
        #slen = np.random.random() * 2 * self.shiftRatio * record.num_frames - record.num_frames * self.shiftRatio # shift length [-ratio*len, ratio*len]
        # Version 2: supervised by classification
        selectRatio = np.random.choice(self.shiftRatio, size=1)
        selectIdx, selectRatio = random.choice(list(enumerate(self.shiftRatio)))
        slen = selectRatio * record.num_frames # select from [-0.4,...,0,...,0.4]*len
        totalfrm = record.max_frame # total frames within one video
        slen = int(slen)
        slen = max((record.start_frame)*(-1), slen)
        slen = min((totalfrm-record.end_frame), slen) # shift distance: [-start_frame, max_frame-end_frame]
        sstart = record.start_frame + slen # shifted interval start frame
        send = record.end_frame + slen
        shiftRecord = VideoRecord([record.vidname, sstart, send, record.caption], self.root)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            segment_indices_shift = self._sample_indices(shiftRecord) if self.random_shift else self._get_val_indices(shiftRecord)
        else:
            segment_indices = self._get_test_indices(record)
            segment_indices_shift = self._get_test_indices(shiftRecord)
        tarInfo = tarfile.open(record.path + '.tar')

        #return self._get(record, segment_indices, shiftRecord, segment_indices_shift, tarInfo, slen / totalfrm)
        return self._get(record, segment_indices, shiftRecord, segment_indices_shift, tarInfo, selectIdx)

    def _get(self, record, indices, srecord, sindices, tarInfo, label):
        """
        Loads the frames of a video at the corresponding
        indices.
        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from.
        Returns:
            1) A list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) An integer denoting the video label.
        """
        # Language part (Check the tokenizer application way)
        # Batch tokenization
        #encoded_captions = self.tokenizer( 
        #    list(record.caption), padding=True, truncation=True, max_length=CFG.max_length
        #)
        # single caption tokenization (Move to Collect)
        #encoded_captions = self.tokenizer(record.caption, return_tensors='pt')
        
        #item = {
        #    key: torch.tensor(values[idx])
        #    for key, values in encoded_captions.items()
        #}
        #item = encoded_captions
        item = {}
        item['vidname'] = record.vidname
        item['caption'] = record.caption
        #item['normSlen'] = normSlen
        item['label'] = label
        # Video part
        indices = indices + record.start_frame
        images = list()
        image_indices = list()
        for seg_ind in indices:
            frame_index = int(seg_ind) + 1 # Plue one since our frame starts from idx 1
            frame_index = min(frame_index, record.end_frame) # To avoid invalid index
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(record.path, frame_index, tarInfo)
                images.extend(seg_img)
                image_indices.append(frame_index)
                if frame_index < record.end_frame:
                    frame_index += 1
        
        # shift video
        sindices = sindices + srecord.start_frame
        simages = list()
        simage_indices = list()
        for seg_ind in sindices:
            frame_index = int(seg_ind) + 1
            frame_index = min(frame_index, srecord.end_frame)
            for i in range(self.frames_per_segment):
                seg_img = self._load_image(srecord.path, frame_index, tarInfo)
                simages.extend(seg_img)
                simage_indices.append(frame_index)
                if frame_index < srecord.end_frame:
                    frame_index += 1


        # sort images by index in case of edge cases where segments overlap each other because the overall
        # video is too short for num_segments*frames_per_segment indices.
        # _, images = (list(sorted_list) for sorted_list in zip(*sorted(zip(image_indices, images))))

        if self.transform is not None:
            images = self.transform(images)
            simages = self.transform(simages)

        if self.test_mode is False:
            item['image'] = images.permute(1,0,2,3) # [3, D, H, W]
            item['shiftImg'] = simages.permute(1,0,2,3)
        else: # Test Mode (Dense Extract)
            _T, _C, _H, _W = images.shape
            item['image'] = images.view(_T//self.frames_per_segment, self.frames_per_segment, _C, _H, _W)
            item['image'] = item['image'].permute(0, 2, 1, 3, 4) # [num_clip, 3, frmPerClip, H, W]
        return item

    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``ActivityNetDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])

# Original function
'''
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
'''
def get_transforms():
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
        transforms.Resize(112),  # image batch, resize smaller edge to 299
        transforms.CenterCrop(112),  # image batch, center crop to square 299x299
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return preprocess
