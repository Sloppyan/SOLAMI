# Model Training

![Training](../assets/training.png)

## Stage 1: Tokenizer Training

### Motion Tokenizer

```
cd models/motiongpt

python train.py --cfg ./configs/config_h3d_stage1_local_body_hand_sep_trans.yaml --nodebug
```


### Speech Tokenizer


## Stage 2: Multi-task Pre-training for Modality Alignment

```
cd  models/vla/scripts

bash stage1_pretrain_multinode.sh
```


## Stage 3: Instruction Tuning for Multi-turn Conversation
For instruction tuning, we adopt two options: full parameter finetuning and LoRA finetuning.

```
cd  models/vla/scripts

bash stage2_it_full_deepspeed_multinode.sh
```


