from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import os
import shutil
from skimage.util import random_noise

np.random.seed(42)
import tensorflow_hub as hub
import tensorflow as tf


def read_files(img_dir, lbl_dir):
    lbls_dataset = []
    images = os.listdir(img_dir)
    img_names = []
    for i, image_name in enumerate(tqdm(images)):  # Load images and labels from source
        img_names.append(image_name)
        with open(lbl_dir + '/' + image_name.split('.')[0] + '.txt') as f:
            lbls_dataset.append(f.readlines())
            # lbls_dataset[i][0] = lbls_dataset[i][0].replace('\n', '_').split(
            #     '_')  # remove unwanted characters and split data
    return lbls_dataset, img_names


def find_data(labels_dataset, typ):
    index = []
    for d in range(len(labels_dataset)):
        print(labels_dataset[d][0])
        print(type(labels_dataset[d][0]))
        if int(labels_dataset[d][0][0]) == typ:
            index.append(d)
    if len(index) == 0:
        print('no labels')
    return index


def xml2dim(labels_dataset):
    lbl = labels_dataset[0]  # category
    a = float(labels_dataset[1])  # box center X
    b = float(labels_dataset[2])  # box center Y
    bbox_width = float(labels_dataset[3])  # box width
    bbox_height = float(labels_dataset[4])  # box height
    return lbl, a, b, bbox_width, bbox_height


def create_imgnlbl(labels_len, name_l, img):
    for label_index in range(labels_len):
        new_labels_ = labels_dataset[pic_index][label_index].split(" ")
        lbl, a, b, bbox_width, bbox_height = xml2dim(new_labels_)  # lbl, a, b, bbox_width, bbox_height
        w = int(img.shape[1])
        h = int(img.shape[0])
        x1, y1 = np.round_((a - bbox_width / 2) * w, 0), np.round_((b - bbox_height / 2) * h, 0)
        x2, y2 = np.round_((a + bbox_width / 2) * w, 0), np.round_((b + bbox_height / 2) * h, 0)

        labels_path = Path(f"{labels_directory}")  # labels path
        h, w, _ = img.shape
        x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
        x2, y2 = x2 / w, y2 / h
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        with (labels_path / name_l).open(mode="a") as label_file:
            label_file.write(
                f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )

    return


def create_lblbox(img, lbl, x1, x2, y1, y2):
    h, w, _ = img.shape
    x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
    x2, y2 = x2 / w, y2 / h
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    return [
        [float(lbl), float(x1 + bbox_width / 2), float(y1 + bbox_height / 2), float(bbox_width), float(bbox_height)]]


def shear(labels_len, new_name, img, shx, shy):
    M = np.float32([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
    sheared_img = cv2.warpPerspective(img, M, (int(img.shape[1] * (1 + shx)), int(img.shape[0] * (1 + shy))))
    for label_index in range(labels_len):
        new_labels_ = labels_dataset[pic_index][label_index].split(" ")
        lbl, a, b, bbox_width, bbox_height = xml2dim(new_labels_)  # lbl, a, b, bbox_width, bbox_height
        h, w, _ = img.shape
        x1, y1 = np.round_((a - bbox_width / 2) * w, 0), np.round_((b - bbox_height / 2) * h, 0)
        x2, y2 = np.round_((a + bbox_width / 2) * w, 0), np.round_((b + bbox_height / 2) * h, 0)
        # Affine Transformation
        u1, u2 = int(M[0][0] * x1 + M[0][1] * y1 + M[0][2]), int(M[0][0] * x2 + M[0][1] * y2 + M[0][2])
        v1, v2 = int(M[1][0] * x1 + M[1][1] * y1 + M[1][2]), int(M[1][0] * x2 + M[1][1] * y2 + M[1][2])

        # sheared_img, u1, u2, v1, v2
        x1 = u1
        x2 = u2
        y1 = v1
        y2 = v2

        labels_path = Path(f"{labels_directory}")  # labels path
        h, w, _ = sheared_img.shape
        x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
        x2, y2 = x2 / w, y2 / h
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        with (labels_path / new_name).open(mode="a") as label_file:
            label_file.write(
                f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )

    return sheared_img


def flip(labels_len, new_name, img, mode):
    """
    modes:
        0 = left to right
        1 = up to down
    """
    for label_index in range(labels_len):
        new_labels_ = labels_dataset[pic_index][label_index].split(" ")
        lbl, a, b, bbox_width, bbox_height = xml2dim(new_labels_)  # lbl, a, b, bbox_width, bbox_height
        h, w, _ = img.shape
        if mode == 0:
            a = 1 - a
            flip_img = cv2.flip(img, 1)
        elif mode == 1:
            b = 1 - b
            flip_img = cv2.flip(img, 0)
        else:
            print('mode does not exist')
            return
        x1, y1 = np.round_((a - bbox_width / 2) * w, 0), np.round_((b - bbox_height / 2) * h, 0)
        x2, y2 = np.round_((a + bbox_width / 2) * w, 0), np.round_((b + bbox_height / 2) * h, 0)

        labels_path = Path(f"{labels_directory}")  # labels path
        h, w, _ = flip_img.shape
        x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
        x2, y2 = x2 / w, y2 / h
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        with (labels_path / new_name).open(mode="a") as label_file:
            label_file.write(
                f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )

    return flip_img


def rotate(labels_len, new_name, img, mode):
    """
    modes:
        0 = 90° counterclockwise
        1 = 180°
        2 = 270° counterclowise / 90° clockwise
    """
    rot_img = None
    for label_index in range(labels_len):
        new_labels_ = labels_dataset[pic_index][label_index].split(" ")
        lbl, a, b, bbox_width, bbox_height = xml2dim(new_labels_)  # lbl, a, b, bbox_width, bbox_height
        h, w, _ = img.shape
        if mode == 0:
            rot_img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
            x1, y1 = np.round_((b - bbox_height / 2) * h, 0), np.round_((1 - a - bbox_width / 2) * w, 0)
            x2, y2 = np.round_((b + bbox_height / 2) * h, 0), np.round_((1 - a + bbox_width / 2) * w, 0)
        elif mode == 1:
            rot_img = cv2.rotate(img, cv2.cv2.ROTATE_180)
            a = 1 - a
            b = 1 - b
            x1, y1 = np.round_((a - bbox_width / 2) * w, 0), np.round_((b - bbox_height / 2) * h, 0)
            x2, y2 = np.round_((a + bbox_width / 2) * w, 0), np.round_((b + bbox_height / 2) * h, 0)
        elif mode == 2:
            rot_img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            x1, y1 = np.round_((1 - b - bbox_height / 2) * h, 0), np.round_((a - bbox_width / 2) * w, 0)
            x2, y2 = np.round_((1 - b + bbox_height / 2) * h, 0), np.round_((a + bbox_width / 2) * w, 0)
        else:
            print('mode does not exist')

        labels_path = Path(f"{labels_directory}")  # labels path
        h, w, _ = rot_img.shape
        x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
        x2, y2 = x2 / w, y2 / h
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        with (labels_path / new_name).open(mode="a") as label_file:
            label_file.write(
                f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )

    return rot_img


def rand_erasing(labels_len, new_name, img, mode):
    """
    modes:
        0 = Object-aware Random Erasing (ORE)
        1 = Image-aware Random Erasing (IRE)
        2 = Image and object-aware Random Erasing (I+ORE)
    """
    im = img
    for label_index in range(labels_len):
        new_labels_ = labels_dataset[pic_index][label_index].split(" ")
        lbl, a, b, bbox_width, bbox_height = xml2dim(new_labels_)  # lbl, a, b, bbox_width, bbox_height
        w = int(img.shape[1])
        h = int(img.shape[0])
        x1, y1 = np.round_((a - bbox_width / 2) * w, 0), np.round_((b - bbox_height / 2) * h, 0)
        x2, y2 = np.round_((a + bbox_width / 2) * w, 0), np.round_((b + bbox_height / 2) * h, 0)

        xe, ye = x2, y2
        We, He = 1, 1
        if (mode == 0):
            while not (xe + We <= x2 and ye + He <= y2):
                xe = np.random.choice(range(int(x1), int(x2)))
                ye = np.random.choice(range(int(y1), int(y2)))
                re = np.random.rand() * 0.7  # maximum % of total bbox area
                Se = int((x2 - x1) * (y2 - y1) * np.random.rand())
                He = int(np.round(np.sqrt(Se * re), 0))
                We = int(np.round(np.sqrt(Se / re), 0))
            # Rectangle Ie=(xe,ye,xe+We, ye+He)
            for i in range(ye, ye + He):
                for j in range(xe, xe + We):
                    val = np.random.choice(range(0, 255))
                    im[i][j][0] = val
                    im[i][j][1] = val
                    im[i][j][2] = val
        elif (mode == 1):
            h1 = int(np.round_(np.random.choice(range(0, int(h))) * 0.5, 0))  # maximum % of total image area
            w1 = int(np.round_(np.random.choice(range(0, int(w))) * 0.5, 0))  # maximum % of total image area
            h2 = int(np.round_(np.random.choice(range(h1, int(h))), 0))
            w2 = int(np.round_(np.random.choice(range(w1, int(w))), 0))
            for i in range(h1, h2):
                for j in range(w1, w2):
                    val = np.random.choice(range(0, 255))
                    im[i][j][0] = val
                    im[i][j][1] = val
                    im[i][j][2] = val
        elif mode == 2:
            while not (xe + We <= x2 and ye + He <= y2):
                xe = np.random.choice(range(int(x1), int(x2)))
                ye = np.random.choice(range(int(y1), int(y2)))
                re = np.random.rand() * 0.7  # maximum % of total bbox area
                Se = int((x2 - x1) * (y2 - y1) * np.random.rand())
                He = int(np.round(np.sqrt(Se * re), 0))
                We = int(np.round(np.sqrt(Se / re), 0))
            # Rectangle Ie=(xe,ye,xe+We, ye+He)
            for i in range(ye, ye + He):
                for j in range(xe, xe + We):
                    val = np.random.choice(range(0, 255))
                    im[i][j][0] = val
                    im[i][j][1] = val
                    im[i][j][2] = val
            h1 = int(np.round_(np.random.choice(range(0, int(h))) * 0.5, 0))  # maximum % of total image area
            w1 = int(np.round_(np.random.choice(range(0, int(w))) * 0.5, 0))  # maximum % of total image area
            h2 = int(np.round_(np.random.choice(range(h1, int(h))), 0))
            w2 = int(np.round_(np.random.choice(range(w1, int(w))), 0))
            for i in range(h1, h2):
                for j in range(w1, w2):
                    val = np.random.choice(range(0, 255))
                    im[i][j][0] = val
                    im[i][j][1] = val
                    im[i][j][2] = val
        else:
            print('ode does not exist')


        labels_path = Path(f"{labels_directory}")  # labels path
        h, w, _ = im.shape
        x1, y1 = x1 / w, y1 / h  # escalate x and y (0 to 1)
        x2, y2 = x2 / w, y2 / h
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        with (labels_path / new_name).open(mode="a") as label_file:
            label_file.write(
                f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
            )

    return im


image_directory = 'test_aug/images/train'
labels_directory = 'test_aug/labels/train'
labels_dataset, image_names = read_files(image_directory, labels_directory)

# pic_index = 0
for pic_index in range(len(image_names)):
    img = np.array(cv2.imread(image_directory + '/' + image_names[pic_index]))
    single_pic_name = image_names[pic_index].split('.')[0]

    """增强"""
    img_gnoise = (255 * random_noise(img, mode='gaussian', var=0.05 ** 2)).astype(np.uint8)
    cv2.imwrite(image_directory + "/" + single_pic_name + '_GN' + '.jpg', img_gnoise)
    name_l = single_pic_name + '_GN' + '' + ".txt"
    create_imgnlbl(len(labels_dataset[pic_index]), name_l, img_gnoise)

    new_name = single_pic_name + '_sheared' + '' + ".txt"
    img_shx = shear(len(labels_dataset[pic_index]), new_name, img, 0.1, 0)  # sheared_img, u1, u2, v1, v2 _SX
    cv2.imwrite(image_directory + "/" + single_pic_name + '_sheared' + '.jpg', img_shx)

    new_name = single_pic_name + '_flip' + '' + ".txt"
    img_fliplr = flip(len(labels_dataset[pic_index]), new_name, img, 0)  # flip_img, x1, x2, y1, y2 _LR
    cv2.imwrite(image_directory + "/" + single_pic_name + '_flip' + '.jpg', img_fliplr)

    new_name = single_pic_name + '_img_r90' + '' + ".txt"
    img_r90 = rotate(len(labels_dataset[pic_index]), new_name, img, 0)  # rot_img, x1, x2, y1, y2 _R90
    cv2.imwrite(image_directory + "/" + single_pic_name + '_img_r90' + '.jpg', img_r90)

    # new_name = single_pic_name + '_erasing' + '' + ".txt"
    # img_re = rand_erasing(len(labels_dataset[pic_index]), new_name, img, 1)  # img _RE
    # cv2.imwrite(image_directory + "/" + single_pic_name + '_erasing' + '.jpg', img_re)

