from dataclasses import dataclass
import torch

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    #feats: torch.tensor
    clips: tuple 
    queries: torch.tensor
    wordlens: torch.tensor

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        #self.feats = self.feats.to(device)
        self.clips = [clip.to(device) for clip in self.clips]
        self.queries = self.queries.to(device)
        self.wordlens = self.wordlens.to(device)
        return self
    

