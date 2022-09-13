#
#python train.py --workers 8 --device 0 --batch-size 12 \
#  --data data/v2x.yaml --img 640 640 --cfg cfg/training/yolov7-tiny-25d.yaml \
#  --weights /home/fy/code/yolov7/runs/train/yolov762/weights/best.pt \
#  --name yolov7 --hyp data/v2x.tiny.yaml


#python train.py --workers 8 --device 1 --batch-size 140 --cache-images --epochs 100 \
#  --weights './runs/train/v2x_v10/weights/best.pt' \
#  --data data/v2x.yaml --img 640 640 --cfg cfg/training/yolov7-tiny-25d.yaml \
#  --name v2x --hyp data/v2x.tiny.yaml



#python train.py --workers 8 --device 0,3 --batch-size 256 --cache-images --epochs 600 \
#  --img 640 640 --cfg cfg/training/yolov7-tiny-25d.yaml \
#  --weights './runs/train/v2x_v5/weights/epoch_124.pt' \
#  --data data/v2x_v.yaml \
#  --name v2x_v --hyp data/v2x.tiny.yaml

python train.py --workers 24 --device 6,7 --batch-size 340 --cache-images --epochs 2000 --save_inters 100 \
  --weights './runs/train/roadside18/weights/last.pt' \
  --data data/roadside.yaml --img 640 640 --cfg cfg/training/yolov7-tiny-25d.yaml \
  --name roadside --hyp data/roadside.tiny.yaml
