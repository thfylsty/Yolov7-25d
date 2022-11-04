import cv2
import json
import numpy as np
import os
from ddd_utils import draw_box_3d, project_to_image, compute_box_3d, rot_y2alpha, alpha2rot_y, add_bird_view,add_bird_view2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def get_label(bboxs_2d, bboxs_3d,classes,truncated_state,occluded_state):
    label = []
    for i in range(len(bboxs_2d)):
        class_id = classes[i]
        truncated = truncated_state[i]
        occluded = occluded_state[i]
        alpha = 0
        class_id = classes[i]
        bbox_2d = [bboxs_2d[i]["xmin"],bboxs_2d[i]["ymin"],bboxs_2d[i]["xmax"],bboxs_2d[i]["ymax"]]
        bbox_3d = bboxs_3d[i]["dim"]
        location = bboxs_3d[i]["location"][0]
        rotation = bboxs_3d[i]["rotation_y"]


        label.append([class_id,truncated,occluded,alpha]+bbox_2d+bbox_3d+location+[rotation])

    # if len(labels)>0:
    #     print(len(labels[0]),labels)
    # col1 ： 类标签
    # col2-4: truncated, occluded, alpha
    # col5-8 ：2D bbox，左上和右下坐标
    # col9-11 : 3D bbox dim 高、宽、长
    # col12-14 ：3D bbox centor 坐标
    # col15: 车体方向角
    return label

def save_date(calib, labels,filename,rotation):

    calib = calib.reshape([-1])
    calib_path = "../v2x/calib/%s.txt" % filename
    label_path = "../v2x/label/%s.txt" % filename
    if os.path.exists(calib_path):
        os.remove(calib_path)
    if os.path.exists(label_path):
        os.remove(label_path)
    with open(calib_path, "a") as f:
        s = "P0: "
        for c in calib:
            s = s + str(c) + " "
        f.write(s)
        s = "\nP1: "
        for cc in rotation:
            for c in cc:
                s = s + str(c) + " "
        f.write(s)

    with open(label_path, "a") as f:
        for label in labels:
            s = ""
            for l in label:
                s = s + str(l) + " "
            s = s + "\n"
            f.write(s)

    # print("save success :", file_idx)
def trans_unifycam(location):

    calib_trans = np.dot(np.linalg.pinv(calib_unify), calib_this)
    location = np.concatenate(
        [location, np.ones((location.shape[0], 1), dtype=np.float32)], axis=1)
    location = np.dot(calib_trans, location.T)
    location = location[:3].T
    location = location.tolist()
    return location


R_lu2k = np.array([[0, -1, 0, ], [0, 0, -1, ], [1, 0, 0, ]])
R_k2lu = np.array([[0, 0, 1, ],
                   [-1, 0, 0, ],
                   [0, -1, 0,]] )
# R_k2lu = np.array([[0, 0, 1, 0],
#                    [-1, 0, 0, 0],
#                    [0, -1, 0, 0],
#                    [0, 0, 0, 1]])
calib_roadside = [[4.3808785552295910e+02, 0.000000e+00, 6.0585738792418056e+02,0.0],
      [0.000000e+00, 4.3608672016711375e+02, 3.9094842550230072e+02,0.0],
      [0.000000e+00, 0.000000e+00, 1.000000e+00,0.0]]
# calib_roadside = np.array(calib_roadside)
calib_v2x = [[2186.359688 ,0.0, 968.712906, 0.0],
               [ 0.0, 2332.160319, 542.356703 ,0.0] ,
               [0.0 ,0.0, 1.0, 0]]
calib_unify = np.array(calib_v2x)


flag = False
imshow=True
# imshow=False
# save = True
save = False
v2x_type = "v2x_v"




class_name = ["Car","Truck","Van","Bus","Pedestrian","Cyclist","Tricyclist","Motorcyclist","Barrowlist","Trafficcone"]
class_id = {j:i for i,j in enumerate(class_name)}
print(class_id)

if v2x_type == "v2x_i":
    files = os.listdir("./cooperative-vehicle-infrastructure-infrastructure-side-image")
else:
    files = os.listdir("./cooperative-vehicle-infrastructure-vehicle-side-image")
files.sort(key=lambda x: float(x[:-4]))

for file in tqdm(files[8100:]):
# if 1:
#     file = "000125.jpg"
#     file = "000009.jpg"
#     print(file)
    bboxs_3d=[]
    bboxs_2d=[]
    occluded_state = []
    truncated_state = []
    classes = []
    # print(file)
    filename = file[:-4]
    if v2x_type == "v2x_i":
        camera_intrinsic_json_path = "./cooperative-vehicle-infrastructure/infrastructure-side/calib/camera_intrinsic/%s.json" % (filename)
        label_josn_path = "./cooperative-vehicle-infrastructure/infrastructure-side/label"
        label_camera_json_path = label_josn_path + "/camera/%s.json" % (filename)
        img_path = "./cooperative-vehicle-infrastructure-infrastructure-side-image/%s.jpg" % (filename)
        calib_lidar2cam_path = "./cooperative-vehicle-infrastructure/infrastructure-side/calib/virtuallidar_to_camera/%s.json" % (filename)
    else:
        camera_intrinsic_json_path = "./cooperative-vehicle-infrastructure/vehicle-side/calib/camera_intrinsic/%s.json" % (filename)
        label_josn_path = "./cooperative-vehicle-infrastructure/vehicle-side/label"
        label_camera_json_path = label_josn_path + "/camera/%s.json" % (filename)
        img_path = "./cooperative-vehicle-infrastructure-vehicle-side-image/%s.jpg" % (filename)
        calib_lidar2cam_path = "./cooperative-vehicle-infrastructure/vehicle-side/calib/lidar_to_camera/%s.json" % (filename)
    if imshow:
        img = cv2.imread(img_path)
    label_camera_json = json.load(open(label_camera_json_path))
    # label_lidar_json = json.load(open(label_lidar_json_path))
    camera_intrinsic_json = json.load(open(camera_intrinsic_json_path))
    calib_lidar2cam = json.load(open(calib_lidar2cam_path))
    # print(calib_lidar2cam)
    K = np.array(camera_intrinsic_json["cam_K"]).reshape([3, 3])
    # K = np.dot(K,np.array([[0.854463098610578, -0.5195105091837793, 0.0012102751176149926], [0.5195000561482762, 0.8544244218454334, -0.0088276425335725], [0.0035518987605347592, 0.008171634487169012, 0.9999599188996214]]))
    # K = np.dot(K,np.array([[-0.0638033225610772, -0.9910914864003576, -0.04429948490729328], [-0.2102873406178483, 0.043997692433495696, -0.7987692871343754], [0.97575114561348, -0.06031492538699515, -0.17158543199893228]]))
    # rotation = np.array([[-0.0638033225610772, -0.9910914864003576, -0.04429948490729328],
    #                      [-0.2102873406178483, 0.043997692433495696, -0.7987692871343754],
    #                      [0.97575114561348, -0.06031492538699515, -0.17158543199893228]])
    rotation = calib_lidar2cam["rotation"]
    translation = calib_lidar2cam["translation"]
    translation = np.dot(np.linalg.pinv(rotation), translation)

    calib_this = np.c_[K, np.zeros(3)]

    # for label in label_camera_json:
    # label = label_camera_json[0]
    if imshow:
        img_s = img.copy()
        bird_view = np.ones((384, 384, 3), dtype=np.uint8) * 230

    # for label in label_lidar_json:
    # print(label_camera_json[0])
    # print(label_lidar_json[0])
    bird_view_c = bird_view.copy()
    for label in label_camera_json:

        # print(label)
        bboxs_2d.append(label["2d_box"])
        classes.append(class_id[label["type"]])
        truncated_state.append(label["truncated_state"])
        occluded_state.append(label["occluded_state"])

        dim = [label["3d_dimensions"][l] for l in label["3d_dimensions"]]
        location = [float(label["3d_location"][l]) for l in label["3d_location"]]

        location[0] += translation[0]
        location[1] += translation[1]
        location[2] += translation[2]

        # x, y, z = -y, -z, x  # 转kitti
        # print(location)
        location = [-location[1], -location[2], location[0]]
        # location = np.dot(R_k2lu,location)

        # dim = [dim[1], dim[0], dim[2]]
        dim = [dim[0], dim[2], dim[1]]

        # array([0.19131181, 6.1437538, -7.27060406])
        rotation_y = -label["rotation"]
        # print(rotation_y)
        if label["type"] == "Trafficcone":
            rotation_y = 0
            # print(label["type"],rotation_y)
        # print(rotation_y)
        # rotation_y = 0
        # location = np.dot(rotation,location)

        location[1] += dim[0] / 2

        # rotation = np.dot(R_lu2k, rotation)
        # print(dim, location, rotation_y)



        # # # 相机坐标系转换 cam --> unify_cam
        #####location = np.array(location).T
        # location[2][0] = 300
        # print(location)
        # location = np.array(location).T
        location = np.dot(R_k2lu, np.array(location)).T
        location = np.dot(rotation, location.transpose(1, 0)).transpose(1, 0)
        location = trans_unifycam(location)
        # print(location)
        # print("===")
        # print(dim)
        # print("dim",label["type"] ,dim)
        dim = np.dot(R_k2lu, np.array(dim)).T
        dim = [dim[2], dim[0], dim[1]]
        dim = np.dot(rotation, np.array(dim).T).T

        dim = trans_unifycam(np.array([dim]))[0]
        dim = [abs(dim[1]), abs(dim[2]), abs(dim[0])]
        # dim = [dim[1], dim[2], dim[0]]
        # print("dim",label["type"], dim)
        # dim[1] += 4
        # dim = [dim[0],dim[1],dim[2]]
        # dim = [dim[0],dim[1],dim[2]]
        # print(dim)


        calib = calib_unify

        # calib = calib_this

        if imshow:
            # print("dim, location, rotation_y,",dim, location, rotation_y)
            box_3d = compute_box_3d(dim, location, rotation_y)

            # # 俯仰角 需要注释坐标系转换
            # box_3d = np.dot(R_k2lu, box_3d.T).T
            # box_3d = np.dot(rotation, box_3d.transpose(1, 0)).transpose(1, 0)

            box_2d = project_to_image(box_3d, calib)
            # print(box_2d)
            txt_show = label["type"]
            img_s = draw_box_3d(img_s, box_2d,txt=txt_show)

            img_bird = add_bird_view(bird_view, dim, location, rotation_y)
            img_bird_c = add_bird_view2(bird_view_c, dim, location, rotation_y)
            # img_bird2 = add_bird_view(bird_view.copy(), dim, location, rotation_y,world_size=60)
            # print(location.shape)
            # bboxs_3d.append([dim, location, rotation_y])

        bboxs_3d.append({"dim":dim, "location":location, "rotation_y":rotation_y})
        # print("dim, location, rotation_y",dim, location, rotation_y)
        # if key == ord("s"):
    if save:
        labels = get_label(bboxs_2d, bboxs_3d, classes, truncated_state, occluded_state)
        save_date(calib, labels,filename,rotation)
    # print(bboxs_3d)
    if imshow:
        cv2.imshow("f", img_s)
        cv2.imshow("img_bird", img_bird)
        cv2.imshow("bird_view_c", bird_view_c)
        cv2.waitKey(1)
        if flag == True:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(0)

        if key == ord("d"):
            flag = False
            continue
        if key == ord("q"):
            exit()
        if key == ord("p"):
            cv2.waitKey(0)
        if key == ord("m"):
            flag = True
