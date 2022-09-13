import os
import random
# path = "/media/fy/Data/DataSet/dataset/v2x_yolo/"

path = "/data/common/datasets/roadside_yolo/"
split_train_val_test = [0.93,0.02,0.05]


image_paths = os.listdir(path+"images")
random.shuffle(image_paths)
total_num = len(image_paths)

split_train_val_test_num = [int(i*total_num) for i in split_train_val_test]

train_list = image_paths[:split_train_val_test_num[0]]
val_list = image_paths[split_train_val_test_num[0]:-split_train_val_test_num[2]]
test_list = image_paths[-split_train_val_test_num[2]:]

print("total_num",total_num)
print(len(train_list))
print(len(val_list))
print(len(test_list))


with open(path + "./train.txt", "w") as f:
    for p in train_list[:-1]:
        f.writelines("./images/" + p + "\n")
    f.writelines("./images/" + train_list[-1])

with open(path + "./val.txt", "w") as f:
    for p in val_list[:-1]:
        f.writelines("./images/" + p + "\n")
    f.writelines("./images/" + val_list[-1])

with open(path + "./test.txt", "w") as f:
    for p in test_list[:-1]:
        f.writelines("./images/" + p + "\n")
    f.writelines("./images/" + test_list[-1])