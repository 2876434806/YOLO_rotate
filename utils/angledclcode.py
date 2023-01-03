import numpy as np
import math
def get_code_len(class_range, mode=0):
    # 获得编码长度
    # class_range : angle_range/omega
    # mode: 0 : binary label, 1: gray label
    # return : 编码长度
    if mode in [0, 1]:
        return math.ceil(math.log(class_range, 2))
    else:
        raise Exception('Only support binary, gray coded label')

def get_all_binary_label(num_label, class_range):
    # 编码所有十进制
    # num_label : angle_range / omega, 90 / omega or 180 / omega
    # class_range : angle_range / omega, 90 / omega or 180 / omega
    # return : all binary label
    all_binary_label = []
    coding_len = get_code_len(class_range)
    tmp = 10 ** coding_len
    for i in range(num_label):
        binay = bin(i)
        binay = int(binay.split('0b')[-1]) + tmp
        binay = np.array(list(str(binay)[1:]), np.int32)
        all_binary_label.append(binay)
    return np.array(all_binary_label)

def binary_label_encode(angle_label, angle_range, omega=1.):
    # 二值码编码
    # angle_label : 十进制gt
    # angle_range : 90 or 180
    # omega : angle discretization granularity
    # return : 十进制对应的二进制
    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_range /= omega
    angle_range = int(angle_range)

    angle_label = np.array(np.round(angle_label), np.int32)
    inx = angle_label == angle_range
    angle_label[inx] = 0
    all_binary_label = get_all_binary_label(angle_range, angle_range)
    binary_label = all_binary_label[angle_label]
    return np.array(binary_label, np.float32)

def angle_label_encode(angle_label, angle_range, omega=1., mode=0):
   # 编码gt_label为DCL
   # angle_label ： 范围在(0,90] 或者 (0,180]
   # angle_range :  90或180
   # omega ： angle discretization granularity
   # mode ： 0: binary label, 1: gray label
   # return ： binary/gray label
    if mode == 0:
        angle_binary_label = binary_label_encode(angle_label, angle_range, omega)
        return angle_binary_label
    elif mode == 1:
        angle_gray_label = gray_label_encode(angle_label, angle_range, omega)
        return angle_gray_label
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')

def binary_label_decode(binary_label, angle_range, omega=1.):
    # 二值码解码
    # binary_label : binary label
    # angle_range : 90 or 180
    # omega : angle discretization granularity
    # return : angle label
    angle_range /= omega
    angle_range = int(angle_range)
    angle_label = np.array(np.round(binary_label), np.int32)
    angle_label = angle_label.tolist()
    all_angle_label = []
    str_angle = ''
    for i in angle_label:
        decode_angle_label = int(str_angle.join(map(str, i)), 2)
        decode_angle_label = 0 if decode_angle_label == 0 else decode_angle_label
        decode_angle_label = decode_angle_label \
            if 0 <= decode_angle_label < int(angle_range) \
            else decode_angle_label - int(angle_range / 2)
        all_angle_label.append(decode_angle_label * omega)
    return np.array(all_angle_label, np.float32)

def get_all_gray_label(angle_range):
    """
    Get all gray label

    :param angle_range: 90/omega or 180/omega
    :return: all gray label
    """
    coding_len = get_code_len(angle_range)
    return np.array(get_grace(['0', '1'], 1, coding_len))


def get_grace(list_grace, n, maxn):

    if n >= maxn:
        return list_grace
    list_befor, list_after = [], []
    for i in range(len(list_grace)):
        list_befor.append('0' + list_grace[i])
        list_after.append('1' + list_grace[-(i + 1)])
    return get_grace(list_befor + list_after, n + 1, maxn)


def gray_label_encode(angle_label, angle_range, omega=1.):
    """
    Encode angle label as gray label

    :param angle_label: angle label, range in [-90,0) or [-180, 0)
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: gray label
    """

    assert (angle_range / omega) % 1 == 0, 'wrong omega'

    angle_label = np.array(angle_label, np.int32)
    angle_label = np.divide(angle_label, omega)
    angle_range /= omega
    angle_range = int(angle_range)

    angle_label = np.array(np.round(angle_label), np.int32)
    inx = angle_label == angle_range
    angle_label[inx] = 0
    all_gray_label = get_all_gray_label(angle_range)
    gray_label = all_gray_label[angle_label]
    return np.array([list(map(int, ''.join(a))) for a in gray_label], np.float32)


def gray_label_decode(gray_label, angle_range, omega=1.):
    """
    Decode gray label back to angle label

    :param gray_label: gray label
    :param angle_range: 90 or 180
    :param omega: angle discretization granularity
    :return: angle label
    """
    angle_range /= omega
    angle_range = int(angle_range)
    angle_label = np.array(np.round(gray_label), np.int32)
    angle_label = angle_label.tolist()
    all_angle_label = []
    all_gray_label = list(get_all_gray_label(angle_range))
    str_angle = ''
    for i in angle_label:
        decode_angle_label = all_gray_label.index(str_angle.join(map(str, i)))
        decode_angle_label = 0 if decode_angle_label == 0 else decode_angle_label
        decode_angle_label = decode_angle_label \
            if 0 <= decode_angle_label < int(angle_range) \
            else decode_angle_label - int(angle_range / 2)
        all_angle_label.append(decode_angle_label * omega)
    return np.array(all_angle_label, np.float32)




def angle_label_decode(angle_encode_label, angle_range, omega=1., mode=0):

    # 解码DCL为gt_label
    # angle_label ： 范围在(0,90] 或者 (0,180]
    # angle_range :  90或180
    # omega ： angle discretization granularity
    # mode ： 0: binary label, 1: gray label
    # return ： gt
    if mode == 0:
        angle_label = binary_label_decode(angle_encode_label, angle_range, omega=omega)
    elif mode == 1:
        angle_label = gray_label_decode(angle_encode_label, angle_range, omega=omega)
    else:
        raise Exception('Only support binary, gray and dichotomy coded label')
    return angle_label

if __name__ == "__main__":
    # test
    angle = angle_label_encode([180, 120, 80, 60], 180, omega=180 / 256., mode=1)
    label = angle_label_decode(angle, 180, 180 / 256., mode=1)
    print(angle)
    print(label)