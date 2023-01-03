# -*- coding: gbk -*-
from PIL import Image
import os


def Square_Generated (read_file):# ����һ������������������Ҫ��������ͼƬת��
    image = Image.open(read_file)   # ����ͼƬ
    w, h = image.size  # �õ�ͼƬ�Ĵ�С
    # print(w,h)
    new_image = Image.new('RGB', size=(max(w, h), max(w, h)),color=(127, 127, 127))  # �����µ�һ��ͼƬ����Сȡ���������һ�ߣ�color������ͼƬ��������ɫ
    # print(background)
    length = int(abs(w - h))  # һ����Ҫ���ĳ���
    box = (0, 0)  # ����box��
    new_image.paste(image, box)                #�����µ�ͼƬ
    return new_image


# source_path = './pics_GeneratedSqare_Test/'
# save_path = './square_pictureGenerated/'
source_path = 'F:\dataset\HRSC2016\dataaugment\\after augment\images_2\\'     # ����ͼƬ��ŵ�·��
save_path = 'F:\dataset\HRSC2016\pad_augment_data\\after_augment\img_pad\\'           # �²�����������ͼƬ��ŵ�·��
if not os.path.exists(save_path):
    os.mkdir(save_path)

file_names = os.listdir(source_path)          # ��ȡ����ͼƬ������
for i in range(len(file_names)):              # ѭ����������
    img = Square_Generated(source_path + file_names[i]) # ͨ������������ȡ�µ�������ͼƬ
    img.save(save_path+file_names[i],'jpeg')             # ����ͼƬ
    print('number',i)
    print(img)
