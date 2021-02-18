'''
images, boxes, labels, confidences 가지고 오게 해주는 코드
'''

import os
import glob
import torch
import numpy as np
from data.transform import transform
from PIL import Image
import torch.utils.data as data
import cv2

import torchvision
import torch.nn.functional as F

# xml 파일 로드
from xml.etree.ElementTree import parse

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class VOC_Dataset(data.Dataset):
    class_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root="../data/VOC2012", split='TRAIN'):
        super(VOC_Dataset, self).__init__()
        root = os.path.join(root)
        # root = os.path.join(root, split)
        self.img_list = sorted(glob.glob(os.path.join(root, '*/JPEGImages/*.jpg')))
        self.anno_list = sorted(glob.glob(os.path.join(root, '*/Annotations/*.xml')))
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        self.class_dict_inv = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.split = split
        self.img_size = 416
        self.output_shape = (416, 416)
        self.normalized_labels = True

    def __getitem__(self, idx):
        # load img
        # image = Image.open(self.img_list[idx])#.convert('RGB')
        # print(type(image))
        image = cv2.resize(cv2.imread(self.img_list[idx]), self.output_shape)  # numpy
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # tensor

        # square resolution
        image, pad = self.pad_to_square(image)
        _, padded_h, padded_w = image.shape

        # Label
        boxes, labels, is_difficult = self.parse_voc(self.anno_list[idx])

        # convert to tensor
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(is_difficult)  # (n_objects)

        # load img name for string
        # os.path.basename(filename) -> 파일이름만 출력을 한다.
        img_name = os.path.basename(self.anno_list[idx]).split('.')[0]
        img_name_to_ascii = [ord(c) for c in img_name]

        # load img width and height
        # img_width, img_height = float(image.size[0]), float(image.size[1])

        # convert to tensor
        boxes = torch.FloatTensor(boxes)
        targets = torch.zeros(len(boxes), 5)
        targets[:, 1:] = boxes

        for i, boxes in enumerate(targets):
            boxes[0] = i


        labels = torch.LongTensor(labels)
        difficulties = torch.ByteTensor(is_difficult)  # (n_objects)
        img_name = torch.FloatTensor([img_name_to_ascii])
        # additional_info = torch.FloatTensor([img_width, img_height])

        # data augmentation
        # image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, self.split)

        # Targets 값 만들기:
        '''
        targets = []
        for i in range(len(labels)):
            targets.append(boxes[i], labels[i])
        '''

        if self.split == "TEST":
            return image, targets, labels, difficulties, img_name  # for evaluations

        return image, targets, labels, difficulties

    def __len__(self):
        return len(self.img_list)

    def set_image_size(self, img_size):
        self.img_size = img_size

    def pad_to_square(self, image, pad_value=0):
        _, h, w = image.shape

        difference = abs(h - w)

        # (top, bottom) padding or (left, right) padding
        if h <= w:
            top = difference // 2
            bottom = difference - difference // 2
            pad = [0, 0, top, bottom]
        else:
            left = difference // 2
            right = difference - difference // 2
            pad = [left, right, 0, 0]

        # Add padding
        image = F.pad(image, pad, mode='constant', value=pad_value)
        '''
        padding을 주는 함수 F.pad
        pad를 마지막 dim에만 줄 경우(pad_left, pad_right)
        pad를 마지막 2개의 dim에 줄 경우(pad_left, pad_right, pad_top, pad_bottom)
        pad를 마지막 3개의 dim에 줄 경우(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        '''
        return image, pad

    def parse_voc(self, xml_file_path):
        '''
        <size>: 이미지의 weight, height, depth 정보
            <width>, <height>, <depth>
        <object>: 이미지 속 object의 정보
            <name>
            <bndbox>
                <xmin>, <ymin>, <xmax>, <ymax>
            <difficult>: 인식하기 어려운지
        '''

        # root node 가지고 오기 (xml 문서의 최상단 루트 태그를 가르킴)
        root = parse(xml_file_path).getroot()

        boxes = []
        labels = []
        is_difficult = []

        # iter 함수는 모든 자식에 대해 탐색 가능
        for obj in root.iter("object"):
            # stop 'name' tag
            name = obj.find('./name')
            class_name = name.text.lower().strip()
            labels.append(self.class_dict[class_name])

            # stop to bbox tag
            # obj 하위노드 중 "-" 이름을 가진 첫 번쨰 노드의 값을 가지고 옴
            bbox = obj.find('./bndbox')
            x_min = bbox.find('./xmin')
            y_min = bbox.find('./ymin')
            x_max = bbox.find('./xmax')
            y_max = bbox.find('./ymax')

            # from str to int

            # x_min = float(x_min.text)
            # y_min = float(y_min.text)
            # x_max = float(x_max.text)
            # y_max = float(y_max.text)
            x_min = float(x_min.text) - 1
            y_min = float(y_min.text) - 1
            x_max = float(x_max.text) - 1
            y_max = float(y_max.text) - 1
            boxes.append([x_min, y_min, x_max, y_max])

            # is_difficult
            is_difficult_str = obj.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def collate_fn(self, batch):
        images = list()
        targets = list()
        difficulties = list()
        labels = list()
        #images, targets, difficulties, _ = list(zip(*batch))

        bboxes = list()

        for b in batch:
            images.append(b[0])
            x = len(images[0][0])
            y = len(images[0][1])
            targets.append(b[1])

            for t in range(len(targets[0])):
                for j in range(5):
                    if j == 0:
                        b[1][t][j] = int(b[1][t][j])
                    else:
                        if j % 2 == 1:
                            b[1][t][j] = float(b[1][t][j] / y)
                        if j % 2 == 0:
                            b[1][t][j] = float(b[1][t][j] / x)

            bboxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        return images, bboxes, labels, difficulties


if __name__ == "__main__":
    '''
    boxes:[tensor([[0.2500, 0.2740, 0.7920, 0.6477],
            [0.6060, 0.3096, 0.7340, 0.4342],
            [0.5740, 0.6370, 0.6100, 0.8114],
            [0.9120, 0.6690, 0.9480, 0.8434]])]
    labels:[tensor([ 0,  0, 14, 14])]
    shape: torch.Size([1, 3, 416, 416])
    '''

    train_set = VOC_Dataset("../data/VOC2012", split='TRAIN')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, collate_fn=train_set.collate_fn,
                                               shuffle=False, num_workers=0, pin_memory=True)

    # 17125개의 Trainloader
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        if i < 3:
            print("images: {}, boxes: {}, labels: {}".format(images, boxes, labels))
            # images: [1, 3, 416, 416]
            # boxes: [[0.0127, 0.4670, 0.2900, 0.6960]]
            # label: [14]
            # print("targets:{}\n".format(targets))
            # list, tuple, tensor
            # print(type(targets), type(targets[0]), type(targets[0][0]))
            # print(targets[0][0])
            # print("++++++++")
            # print("images:{}\n boxes:{}\n labels:{}\n".format(images, boxes, labels))
            # print("shape\n", images.shape)
            # print("shape\n", boxes.size())
            # print("shape\n", labels.shape)
            print("=====================")