gpus=4,5,6,7
gpun=4
master_addr=127.0.0.3
master_port=29502
# find all configs in configs/
model=howto

# ------------------------ need not change -----------------------------------
config_file=configs/$model\.yaml
output_dir=outputs/$model

# Multiple GPUs
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun \
--master_addr $master_addr \
--master_port $master_port \
train_net.py \
--skip-test \
--config-file $config_file \
OUTPUT_DIR $output_dir \
SOLVER.BATCH_SIZE 32 \
TEST.BATCH_SIZE 4 \
SOLVER.TEST_PERIOD -1 \
DATALOADER.NUM_WORKERS 4 \
INPUT.NUM_SEGMENTS 5 \
MODEL.CLIP.LOSS.SHIFTWEIGHT 1 \
MODEL.CLIP.LOSS.FGWEIGHT 10 \
MODEL.CLIP.LOSS.FGPOSWEIGHT 1000 \
SOLVER.LR 0.001