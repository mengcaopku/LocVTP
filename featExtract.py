import argparse
import os

import torch

from clip.config import cfg
from clip.data import make_data_loader
from clip.engine.inference import inference
from clip.modeling import build_model
from clip.utils.checkpoint import ClipCheckpointer
from clip.utils.comm import synchronize, get_rank
from clip.utils.logger import setup_logger
import h5py
import time

def main():
    parser = argparse.ArgumentParser(description="Clip")
    parser.add_argument(
        "--config-file",
        default="configs/2dtan_128x128_pool_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--feat_dir", type=str, help="The output feature path.")
    parser.add_argument("--feat_name", type=str, default='c3d_features.hdf5', help="The name of output features")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("clip", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = ClipCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    # Set output file
    #if args.local_rank == 0:
    #f = h5py.File(os.path.join(args.feat_dir, args.feat_name), 'w')
    f = h5py.File(os.path.join(args.feat_dir, "{}_{}".format(str(args.local_rank), args.feat_name)), 'w')

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for dataset_name, data_loader_val in zip(dataset_names, data_loaders_val):
        for iteration, (clip, _, capTokens, vidnames, _, _) in enumerate(data_loader_val):
            with torch.no_grad():
                for vidnam, vidClip in zip(vidnames, clip):
                    print("{} --- {}: {}".format(args.local_rank, vidnam, vidClip.shape))
                    feat = []
                    for eachClip in vidClip:
                        eachClip = eachClip.unsqueeze(0).cuda() # [1, 3, 16, 112, 112]
                        clipEmbed = model.vidProjector(model.videoEncoder(eachClip))
                        clipEmbed = clipEmbed.cpu() # release GPU memory
                        feat.append(clipEmbed) 
                    features = torch.cat(feat, 0)
                    features = features.detach().numpy()

                    fgroup = f.create_group(vidnam)
                    fgroup.create_dataset('c3d_features', data=features)
                    #fgroup.create_dataset('total_frames', data=np.array(total_frames))
                    #fgroup.create_dataset('valid_frames', data=np.array(valid_frames))
                    #print('{}/{}: {} has been processed...'.format(idx, video_list_len, video_name))

                    synchronize()

if __name__ == "__main__":
    main()
