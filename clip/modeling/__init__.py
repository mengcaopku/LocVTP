from .clip import CLIP
ARCHITECTURES = {"CLIP": CLIP}

def build_model(cfg):
    return ARCHITECTURES[cfg.MODEL.ARCHITECTURE](cfg)
