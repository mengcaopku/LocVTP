import os

from yacs.config import CfgNode as CN
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.ARCHITECTURE = "CLIP"
_C.MODEL.WEIGHT = ""

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.NUM_SEGMENTS = 1
_C.INPUT.FRAMES_PER_SEGMENT = 16
_C.INPUT.SHIFTRATIO = []

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.NUM_CLIPS = 128

_C.MODEL.CLIP.VIDEOEMBED = CN()
_C.MODEL.CLIP.VIDEOEMBED.NBCLASS = 487
_C.MODEL.CLIP.VIDEOEMBED.LAYER = 6
#_C.MODEL.CLIP.VIDEOEMBED.MODEL = "C3D"
_C.MODEL.CLIP.VIDEOEMBED.PRETRAIN = True
_C.MODEL.CLIP.VIDEOEMBED.TRAINABLE = True 
_C.MODEL.CLIP.VIDEOEMBED.DIM = 4096

_C.MODEL.CLIP.TEXTEMBED = CN()
_C.MODEL.CLIP.TEXTEMBED.MODEL = "distilbert-base-uncased"
_C.MODEL.CLIP.TEXTEMBED.PRETRAIN = True
_C.MODEL.CLIP.TEXTEMBED.TRAINABLE = False 
_C.MODEL.CLIP.TEXTEMBED.DIM = 768

_C.MODEL.CLIP.PROJECTION = CN()
_C.MODEL.CLIP.PROJECTION.DIM = 256

_C.MODEL.CLIP.FEATPOOL = CN()
_C.MODEL.CLIP.FEATPOOL.INPUT_SIZE = 4096
_C.MODEL.CLIP.FEATPOOL.HIDDEN_SIZE = 512
_C.MODEL.CLIP.FEATPOOL.KERNEL_SIZE = 2

_C.MODEL.CLIP.FEAT2D = CN()
_C.MODEL.CLIP.FEAT2D.POOLING_COUNTS = [15,8,8,8]

_C.MODEL.CLIP.INTEGRATOR = CN()
_C.MODEL.CLIP.INTEGRATOR.QUERY_HIDDEN_SIZE = 512
_C.MODEL.CLIP.INTEGRATOR.LSTM = CN()
_C.MODEL.CLIP.INTEGRATOR.LSTM.NUM_LAYERS = 3
_C.MODEL.CLIP.INTEGRATOR.LSTM.BIDIRECTIONAL = False

_C.MODEL.CLIP.PREDICTOR = CN() 
_C.MODEL.CLIP.PREDICTOR.HIDDEN_SIZE = 512
_C.MODEL.CLIP.PREDICTOR.KERNEL_SIZE = 5
_C.MODEL.CLIP.PREDICTOR.NUM_STACK_LAYERS = 8

_C.MODEL.CLIP.LOSS = CN()
_C.MODEL.CLIP.LOSS.MIN_IOU = 0.3
_C.MODEL.CLIP.LOSS.MAX_IOU = 0.7
_C.MODEL.CLIP.LOSS.TEMPERATURE = 1.0
_C.MODEL.CLIP.LOSS.LOSSTYPE = 'L1' # {'L1', 'SL1', 'MSE', 'CrossEntropy'}
_C.MODEL.CLIP.LOSS.SHIFTWEIGHT = 1
_C.MODEL.CLIP.LOSS.FGWEIGHT = 1
_C.MODEL.CLIP.LOSS.FGPOSWEIGHT = 1000

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 12
_C.SOLVER.LR = 0.01
_C.SOLVER.CHECKPOINT_PERIOD = 1
_C.SOLVER.TEST_PERIOD = 1
_C.SOLVER.BATCH_SIZE = 32
_C.SOLVER.MILESTONES = (8, 11)

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.NMS_THRESH = 0.4
 
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
