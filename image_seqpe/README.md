## Install

```bash
conda env create -f conda_swin.yml
```

## Data

```bash
# please move the data to ./datasets/imagenet
bash scripts/prepare_imagenet.py
```


## Eval

> Please download the ckpt from https://huggingface.co/huayangli/seqpe/.
```
bash eval.sh $CKPT_DIR/image_seqpe_ckpt
```


## Train
```bash
bash runs/ours_vit.sh -n 4
```


