# FastCLIP

```
python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --rdzv-id=4204 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.1' src/training/main.py --save-frequency 1 --train-data './datasets/dfn_data/00000{000..139}.tar' --train-num-samples 1000000 --data_size 1400000 --warmup 500 --batch-size 320 --epochs 30 --workers 6 --model ViT-B-16 --name fastclipv3_dive9_v1 --seed 2025 --wd 0.2 --local-loss --fastclip --multiply_tau --temperature_scheme global_learnable --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 30 --report-to tensorboard
```

![alt text](tb/vanilla/train.png)
![alt text](tb/vanilla/eval.png)

```
train_output_dir='./logs/fastclipv3_dive9_v1'
data_dir='./datasets/datacomp'
arch='ViT-B-16'
epoch=18

CUDA_VISIBLE_DEVICES=6 python ./datacomp/evaluate.py --train_output_dir "${train_output_dir}" --data_dir "${data_dir}" --epoch "${epoch}" --arch "${arch}"
```

```
=== Final results ===
MSCOCO: 0.028415114618837833
ImageNet 1k: 0.08198
```

# OpenAI CLIP

```
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --node_rank=0 --rdzv-id=4206 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.3' src/training/main.py --save-frequency 1 --train-data './datasets/dfn_data/00000{000..139}.tar' --train-num-samples 1000000 --data_size 1400000 --warmup 500 --batch-size 320 --epochs 30 --workers 6 --model ViT-B-32 --pretrained openai --name openai_pretrained_clip --seed 2025 --wd 0.2 --local-loss --fastclip --multiply_tau --temperature_scheme global_learnable --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 30 --report-to tensorboard
```

```
2025-04-14,23:32:09 | INFO | Eval Epoch: 0 mscoco/image_retrieval_recall@1: 0.3042      mscoco/text_retrieval_recall@1: 0.5014  mscoco/image_retrieval_recall@5: 0.5597  mscoco/text_retrieval_recall@5: 0.7498  mscoco/image_retrieval_recall@10: 0.6689        mscoco/text_retrieval_recall@10: 0.8354  mscoco/mean_recall@1: 0.4028    imagenet1k/acc1: 0.6333 imagenet1k/acc5: 0.8881 imagenet1k/mean_per_class_recall: 0.6334
```

![alt text](tb/openAI/eval.png)

# FastCLIP + 0.5 Cross Entropy Distillation

```
CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --rdzv-id=4204 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.1' src/training/main.py --save-frequency 1 --train-data ./datasets/dfn_data/00000{000..139}.tar --datacomp-path ./datasets/datacomp --train-num-samples 1000000 --data_size 1400000 --warmup 500 --batch-size 320 --epochs 30 --workers 6 --model ViT-B-16 --distill-model ViT-B-32 --distill-pretrained openai --distill-mode cross_entropy --distill-weight 0.5 --name fastclip_dist0.5crossEntv0_dive9_v0 --seed 2025 --wd 0.2 --local-loss --fastclip --multiply_tau --temperature_scheme global_learnable --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 30 --report-to tensorboard
```

# FastCLIP + Feature Cross Entropy Distillation

```
CUDA_VISIBLE_DEVICES=8,9 python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --rdzv-id=4204 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.1:29500' src/training/main.py --save-frequency 1 --train-data "./datasets/dfn_data/00000{000..139}.tar" --datacomp-path ./datasets/datacomp --train-num-samples 1000000 --data_size 1400000 --warmup 500 --batch-size 320 --epochs 30 --workers 6 --model ViT-B-16 --distill-model ViT-B-32 --distill-pretrained openai --distill-mode feature --distill-weight 0.5 --name fastclip_dist0.5featurev0_dive9_v0 --seed 2025 --wd 0.2 --local-loss --fastclip --multiply_tau --temperature_scheme global_learnable --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 30 --report-to tensorboard
```