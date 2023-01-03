import os
import shutil
import tqdm

filename = "F:\Code\yolov5_rotation-master\datasets\yolo_hrsc\\train.txt"
label = []
labels = []
with open(filename) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    #专门处理label 换png为txt
    for i in lines:
        i = i.replace('png', 'txt')
        label.append(i)

for i in label:
    i = i.replace('images', 'labels')
    labels.append(i)
print(labels)


# from_path ="F:/Code/yolov5_rotation-master/datasets/yolo_hrsc/images/"
# to_path = "F:/dataset/HRSC2016/pad_augment_data/images/"
from_path ="F:/Code/yolov5_rotation-master/datasets/yolo_hrsc/labels/"
to_path = "F:/dataset/HRSC2016/pad_augment_data/labels/"
files = os.listdir(from_path)
# for i in lines:
for i in labels:
    for j in files:
        # fullname = os.path.join("F:/Code/yolov5_rotation-master/datasets/yolo_hrsc/images/", j)
        fullname = os.path.join("F:/Code/yolov5_rotation-master/datasets/yolo_hrsc/labels/", j)
        if i == fullname:
            print(1)
            target = os.path.join(to_path, j)
            print(target)
            f = open(target, "w")
            f.close()
            shutil.copyfile(fullname, target)


