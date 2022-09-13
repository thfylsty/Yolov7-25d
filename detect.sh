
export CUDA_VISIBLE_DEVICES=7

#python detect.py --weights ./weights/roadside18last.pt  \
# --conf 0.25 --img-size 640 \
# --source /media/fy/Data/DataSet/dataset/roadside/roadside0.mp4 \
# --dataset roadside --bevsize 300 --wordsize 100 --view-img
#
#python detect.py --weights ./weights/v2xv.pt  \
# --conf 0.25 --img-size 640 \
# --source /media/fy/Data/DataSet/dataset/v2xv.mp4 \
# --dataset v2xv --bevsize 300 --wordsize 100 --view-img

#python detect.py --weights ./weights/roadside20last.pt  \
# --conf 0.25 --img-size 640 \
# --source /home/fuyu/data/videos/roadside0.mp4 \
# --dataset roadside --bevsize 300 --wordsize 100

python detect.py --weights ./weights/roadside20last.pt  \
 --conf 0.25 --img-size 640 \
 --source /home/fuyu/data/videos/roadside1.mp4 \
 --dataset roadside --bevsize 300 --wordsize 100

 python detect.py --weights ./weights/roadside20last.pt  \
 --conf 0.25 --img-size 640 \
 --source /home/fuyu/data/videos/roadside2.mp4 \
 --dataset roadside --bevsize 300 --wordsize 100
##
#
#python detect.py --weights ./weights/v2xi.pt  \
# --conf 0.25 --img-size 640 \
# --source /home/fuyu/data/videos/v2xi.mp4 \
# --dataset v2xi --bevsize 500 --wordsize 120
#
#
#python detect.py --weights ./weights/v2xv.pt  \
# --conf 0.25 --img-size 640 \
# --source /home/fuyu/data/videos/v2xv.mp4 \
# --dataset v2xv --bevsize 500 --wordsize 120