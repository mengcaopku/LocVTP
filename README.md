# LocVTP: Video-Text Pre-training for Temporal Localization

PyTorch Implementation of paper:

> **LocVTP: Video-Text Pre-training for Temporal Localization (ECCV 2022)**
>
> Meng Cao, Tianyu Yang, Junwu Weng, Can Zhang, Jue Wang and Yuexian Zou\*.



## Updates

* We released the **uncleaned** code to meet ECCV requirements, i.e. the code should be presented before the camera ready DDL. The cleaned up code will be provided asap.

  

## Data Preparation

* Download [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) and put them in the 'datasets/howto' directory.



## Pre-training

```bash
# 64 V100 GPUs required
master_addr=127.0.0.3
master_port=29502
model=howto

config_file=configs/$model\.yaml
output_dir=outputs/$model

python -m torch.distributed.launch \
--nproc_per_node=$gpun \
--master_addr $master_addr \
--master_port $master_port \
train_net.py \
--skip-test \
--config-file $config_file \
OUTPUT_DIR $output_dir \
SOLVER.BATCH_SIZE 128 \
TEST.BATCH_SIZE 4 \
SOLVER.TEST_PERIOD -1 \
DATALOADER.NUM_WORKERS 4 \
INPUT.NUM_SEGMENTS 5 \
MODEL.CLIP.LOSS.SHIFTWEIGHT 1 \
MODEL.CLIP.LOSS.FGWEIGHT 10 \
MODEL.CLIP.LOSS.FGPOSWEIGHT 1000 \
SOLVER.LR 0.001
```



## Feature Extractor

```bash
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
python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
featExtract.py --config-file $config_file --ckpt $weight_file \
--feat_dir outputs/$model/vidfeats  --feat_name $model"_c3d_features.hdf5" TEST.BATCH_SIZE $batch_size

# 3. Then conduct concatHDF5.py to geneate the final file.
Python concatHDF5.py
```



## ToDo

```markdown
- [ ] More dataset support
- [ ] More backbone support
  - [x] C3D
  - [ ] S3D
  - [ ] ViT + TimesFormer
- [ ] Downstream task support
```

### 

## Other Info

### Citation

Please **[★star]** this repo and **[cite]** the following paper if you feel our work useful to your research:

```
@inproceedings{cao2022locvtp,
    title     = {LocVTP: Video-Text Pre-training for Temporal Localization},
    author    = {Cao, Meng and Yang, Tianyu and Weng, Junwu and Zhang, Can and Wang, Jue and Zou, Yuexian},
    booktitle = {European Conference on Computer Vision},
    year      = {2022}
}

@article{cao2022locvtp,
  title={LocVTP: Video-Text Pre-training for Temporal Localization},
  author={Cao, Meng and Yang, Tianyu and Weng, Junwu and Zhang, Can and Wang, Jue and Zou, Yuexian},
  journal={arXiv preprint arXiv:2207.10362},
  year={2022}
}
```
