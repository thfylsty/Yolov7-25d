#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
NGPUS=6
export OMP_NUM_THREADS=48
export NCCL_DEBUG=info

#
#python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 29501 train.py \
#  --workers 16 --batch-size 340 --cache-images --epochs 3000 --save_inters 100 \
#  --weights './runs/train/roadside19/weights/epoch_1999.pt' \
#  --data data/roadside.yaml --img 640 640 --cfg cfg/training/yolov7-tiny-25d.yaml \
#  --name roadside --hyp data/roadside.tiny.yaml

python -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port 29501 train.py \
  --workers 48 --batch-size 900 --epochs 1000 --save_inters 50 --img 640 640 --test_intervals 10 \
  --weights '' \
  --data data/roadside_2d.yaml --cfg cfg/training/yolov7-tiny.yaml \
  --name roadside2d --hyp data/roadside_2d.tiny.yaml

#-master_addr 127.0.0.2 --master_port 29501 tools/train_net.py
