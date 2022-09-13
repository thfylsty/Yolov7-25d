import cv2
import numpy as np
import os
from tqdm import tqdm
import json
import shutil
from ddd_utils import draw_box_3d, project_to_image, compute_box_3d,rot_y2alpha,alpha2rot_y
print("version",1)
import threading
show_image = True
# show_image = False

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def loadlabel(label_path,calibs,lidar2cam=False):
    with open(label_path) as f:
        labels = json.load(f)
        objs = []
        for label in labels:
            obj = {}
            position = label['psr']['position']
            rotation = label['psr']['rotation']['z']
            scale = label['psr']['scale']
            x, y, z = position["x"], position["y"], position["z"]
            xs, ys, zs = scale["x"], scale["y"], scale["z"]
            obj["obj_type"],obj["obj_id"]  = label['obj_type'],label['obj_id']
            obj["xyz"] = [x, y, z]
            obj["whl"] = [ xs, ys, zs, rotation]
            obj["r"] = rotation
            obj["cvat"] = label["cvat"]

            if lidar2cam:
                obj["location"] = {}
                for cam in ["front", "right", "left"]:
                    location = [x, y, z]
                    location = np.expand_dims(np.concatenate([location, np.ones(1)], 0), 1)
                    p = np.array(calibs[cam]["extrinsic"]).reshape((4, 4))[:3]
                    location = np.dot(p, location)
                    obj["location"][cam] = location.reshape(-1).tolist()
            objs.append(obj)
    return objs


def save_label(objs,label_path):
    filename = label_path.split("/")[-1].split(".")[0]
    label_path = os.path.dirname(label_path)



    for cam in ["front", "right", "left"]:
        file_name = "{}/labels/{}_{}_{}.txt".format(os.path.dirname(label_path),label_path.split("/")[-1], filename, cam)
        with open(file_name, "w") as f:
            for labels in objs:
                class_idx = class_map[labels["obj_type"]]
                xyxy = list(labels["cvat"][cam].values())
                if xyxy[0]==-1:
                    continue
                xyz = labels["location"][cam]
                whl = labels["whl"]
                r = labels["r"]
                label_list = [class_idx] + xyxy + xyz + whl + [r]
                label_txt = "{} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(*label_list)
                if not labels is objs[-1]:
                    label_txt += "\n"
                f.write(label_txt)


def save_calib(calibs,label_path):
    for cam in calibs:
        calib = calibs[cam]
        s = "P0:"
        for c in calib["intrinsic"]:
            s += " "
            s += str(c)
        path_split = label_path.split("/")
        file_name = "{}_c25/training/calib/{}_{}_{}.txt".format(path_split[0], path_split[1], path_split[3].split(".")[0],cam)
        with open(file_name, "w") as f:
            f.write(s)

def save_image(imgs_path,bag_name,img_file,target_pth):

    for cam in ["front","right","left"]:
        source_image_path = imgs_path[cam]
        target_image_path = "{}/images/{}_{}".format(target_pth,bag_name,img_file.replace(".","_{}.".format(cam)))
        shutil.copy(source_image_path,target_image_path)

def readcalib(bag_dir):
    calibs={}
    for cam in ["front","right","left"]:
        with open(os.path.join(bag_dir,"calib/camera/"+cam+".json")) as f:
            calib = json.load(f)
        # calib_ex = calib["extrinsic"]
        # calib_in = calib["intrinsic"]
        calib["intrinsic"].insert(4, 0.0)
        calib["intrinsic"].insert(8, 0.0)
        calib["intrinsic"].insert(12, 0.0)
        calibs[cam]= calib
    return calibs


def init_dir(p,n):
    mkdir("{}/{}/".format(p,n))
    mkdir("{}/{}/images".format(p,n))

def get_imgs_path(img_path):
    imgs_path = {}
    for cam in ["front","right","left"]:
        imgs_path[cam] = img_path.format(cam)
    return imgs_path


# class_map = {"Car":"Car","Pedestrian":"Pedestrian","Truck":"Truck",
#              "Bus":"Bus","Motorcycle":"Motorcyclist","MotorcyleRider":"Motorcyclist",
#              "Bicycle":"Cyclist","BicycleRider":"Cyclist","Van":"Van","Unknown":"Trafficcone"} #,"Van":"van",
#
# class_map = {"Car":"Car","Pedestrian":"Pedestrian","Truck":"Truck",
#              "Bus":"Bus","Motorcycle":"Motorcyclist","MotorcyleRider":"Motorcyclist",
#              "Bicycle":"Cyclist","BicycleRider":"Cyclist","Van":"Van","Unknown":"Trafficcone"} #,"Van":"van",
class_map = {'Pedestrian':0, 'Car':1, 'MotorcyleRider':2, 'Crane':3, 'Motorcycle':4, 'Bus':5, 'BicycleRider':6, 'Van':7, 'Excavator':8, 'TricycleRider':9, 'Truck':10}

def get_label(boxs_2d,boxs_3d):
    # bboxs_3d.append([dim, location, rotation_y, box_2d[-1],class_name])
    # col1 ： 类标签
    # col5-8 ：2D bbox，左上和右下坐标
    # col9-11 : 3D bbox dim 高、宽、长
    # col12-14 ：3D bbox centor 坐标
    # col15: 车体方向角
    # print(boxs_2d)
    labels = []
    for box_3d in boxs_3d:

        class_name = class_map[box_3d[4]]

        bbox_from3d = box_3d[3]
        [bbox_xy0,bbox_xy1] = compare_bbox(boxs_2d,bbox_from3d)
        dim = box_3d[0]
        location = [_[0] for _ in box_3d[1].tolist()]
        rotation = box_3d[2]
        label = [class_name,0,0,0,bbox_xy0[0],bbox_xy0[1],bbox_xy1[0],bbox_xy1[1]]+dim+location+[rotation]
        labels.append(label)

        # print(class_name)

    return labels



out_size = 400
world_size = 80
def project_3d_to_bird( pt):

    pt[0] += world_size / 2
    pt[1] = world_size - pt[1]
    pt = pt * out_size / world_size
    return pt.astype(np.int32)


def showddd(imgs_path,calibs,objs):
    imgs = None
    bird_views = None
    labels_list = []
    for cam in ["left", "front", "right"]:
        bboxs_3d = []
        img = cv2.imread(imgs_path[cam])
        bird_view = np.ones((out_size, out_size, 3), dtype=np.uint8) * 230

        for obj in objs:

            [x, y, z] = obj["xyz"]
            [ xs, ys, zs, r] = obj["whl"]

            z -= zs/2
            rotation_y = r

            dim = [zs, ys, xs]
            class_name = obj["obj_type"]
            #
            location = [x, y, z]
            location = np.expand_dims(np.concatenate([location,np.ones(1)],0),1)
            p = np.array(calibs[cam]["extrinsic"]).reshape((4,4))[:3]
            location = np.dot(p, location)
            # print(class_name, location.T)
            # location = np.array(location)
            #
            calib = np.array(calibs[cam]["intrinsic"]).reshape((3,4))
            box_3d = compute_box_3d(dim, location, rotation_y)
            rect = box_3d[:4, [0, 2]]
            for k in range(4):
                rect[k] = project_3d_to_bird(rect[k])
            lc = (250, 152, 12)
            cv2.polylines(
                bird_view, [rect.reshape(-1, 1, 2).astype(np.int32)],
                True, lc, 2, lineType=cv2.LINE_AA)
            #
            if box_3d.mean(0)[2] <0:
                continue
            box_2d = project_to_image(box_3d, calib)
            # h = max(box_2d[:,1]) - min(box_2d[:,1])
            # w = max(box_2d[:,0]) - min(box_2d[:,0])
            # if abs(h) <10 or abs(w)<10  :
            #     continue
            #     # print("wh",class_name,box_2d[0][0]-box_2d[7][0],box_2d[0][1]-box_2d[7][1])
            # if min(box_2d[-1])<-100 or box_2d[-1][0]>1400 or box_2d[-1][1]>900 :
            #     continue
            img = draw_box_3d(img, box_2d)
            #
            # bboxs_3d.append([dim, location, rotation_y, box_2d,class_name])

            xyxy = list(obj["cvat"][cam].values())
            xyxy = [int(float(i)) for i in xyxy]
            if sum(xyxy) < 0:
                continue
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 255, 0), 2)


        # labels_list.append([labels,cam])
        if show_image:
            if cam=="left":
                imgs = img
            else:
                imgs = np.hstack((imgs,img))
        #
            if cam=="left":
                bird_views = bird_view
            else:
                bird_views = np.hstack((bird_views,bird_view))
    if show_image:
        imgs = cv2.resize(imgs,(imgs.shape[1]//2,imgs.shape[0]//2))
        cv2.imshow("f",imgs)
        cv2.imshow("f2",bird_views)
        cv2.waitKey(1)
    return labels_list

def process(bag_idx,bag_name):
    bag_dir = os.path.join(data_path, dataname, bag_name)
    cvat_label_dir = os.path.join(data_path, cvat_path, bag_name)
    if "2022-08" in cvat_label_dir:
        return
    label_files = os.listdir(cvat_label_dir)
    label_files.sort()
    bar = tqdm(label_files)
    label_num = len(label_files)
    calibs = readcalib(bag_dir)
    for label_file in bar:
        bar.set_description("bag_name bag/all {} {}/{}".format(bag_name, bag_idx + 1, len(bags_name)))
        label_path = os.path.join(cvat_label_dir, label_file)
        img_file = label_file.replace("json", "png")
        images_path = "%s/camera/{}/%s" % (bag_dir, img_file)
        objs = loadlabel(label_path, calibs, lidar2cam=True)
        imgs_path = get_imgs_path(images_path)

        showddd(imgs_path,calibs,objs)

        # save_image(imgs_path, bag_name, img_file, data_path + target_pth)
        # save_label(objs, label_path.replace(cvat_path, target_pth))
        #


def main():

    init_dir(data_path,target_pth)

    for bag_idx,bag_name in enumerate(bags_name):
        process(bag_idx,bag_name)
        # t = threading.Thread(target=process, args=(bag_idx,bag_name))
        # t.start()



if __name__ == '__main__':
    data_path = "/home/fy/dataset/roadside/cvat/cvat114/"
    # data_path = "/data/common/datasets/"
    dataname = "roadside"
    cvat_path = "roadside_concat"
    target_pth = "roadside_yolo"
    bags_name = os.listdir(data_path + dataname)
    main()

