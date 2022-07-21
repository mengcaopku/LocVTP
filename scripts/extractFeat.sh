# find all configs in configs/
model=howto
config_file=configs/$model.yaml
# select weight to evaluate
weight_file=outputs/$model/model_10e.pth
# set your gpu id
gpus=0,1,2,3,4,5,6,7
# number of gpus
gpun=8
# batch size
batch_size=16

master_addr=127.0.0.1
master_port=29501

# 1. Create feature directory
if [ ! -d "outputs/$model/vidfeats" ]; then
      mkdir outputs/$model/vidfeats
fi

# 2. This command generate hdf5 file for each single process
#CUDA_VISIBLE_DEVICES=$gpus /apdcephfs/private_mengcao/anaconda3/envs/detr/bin/python -m torch.distributed.launch \
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
featExtract.py --config-file $config_file --ckpt $weight_file \
--feat_dir outputs/$model/vidfeats  --feat_name $model"_c3d_features.hdf5" TEST.BATCH_SIZE $batch_size

# 3. Then conduct concatHDF5.py to geneate the final file.
# Python concatHDF5.py
