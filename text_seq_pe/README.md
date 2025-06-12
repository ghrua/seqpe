## Install

```bash
conda env create -f conda_text_seqpe.yml
conda activate text_seqpe
$(dirname $(which python))/pip install -e . # ensure the pip is the pip from text_seqpe env
mkdir -p ./text_seq_pe_out
```

## Data
+ Wikitext-103: The data will automatically downloaded using huggingface
+ QA data: https://huggingface.co/huayangli/seqpe/squad_qa


## Eval

> Please download the from ckpts from https://huggingface.co/huayangli/seqpe/
```bash
# wikitext 103
bash eval.sh $CKPT_DIR/lm_seqpe_ckpt

# qa
bash qa_eval.sh $CKPT_DIR/qa_seqpe_ckpt best_model ppl
bash qa_eval.sh $CKPT_DIR/qa_seqpe_ckpt best_model gen
```


## Train
```bash
# wikitext 103
bash runs/ours_gpt2_wt103.sh -n 4

# qa
bash runs/ours_gpt2_qa.sh -a squad_qa -n 4 -P $CKPT_DIR/lm_seqpe_ckpt/ -S 16 -B 16 -Q 2 -e 10000
```
