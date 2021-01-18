'''
utils.py
yolov3을 진행하는 과정에서 필요한 helper 함수들을 정리
- bbox_iou(box1, box2)
    - IoU 계산하여 two bounding boxes 반환
- NMS(prediction, confidence, num_classes, nms_conf)
    - Objectness 점수와 thresholding과 non-maximal suppression과정을 거침
- unique(tensor):
    - 같은 class 내에 여러개의 true detecion이 나올 수 있으므로 unique라는 함수를 만들어 주어진 이미지에 대해 중복되지 않은 class 가져옴
'''
import torch
import numpy as np

def bbox_iou(box1, box2):
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    x1 = torch.max(b1_x1, b2_x1)
    y1 = torch.max(b1_y1, b2_y1)
    x2 = torch.min(b1_x2, b2_x2)
    y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou

# confidence: objectness score threshold
# num_classes: 80 COCO
# nms_conf: NMS IoU threshold

def NMS(prediction, confidence, nms_conf=0.4):

    # prediction = torch.Size([1, 10647, 85])
    # prediction[:,:,4] 의 값은: tensor([[0.5970, 0.6153, 0.5694,  ..., 0.4194, 0.4677, 0.4085]])
    # conf_mask: tensor([[1., 1., 1.,  ..., 0., 0., 0.]])
    # unsqueeze을 통해 torch.Size([1, 10647]) 가 torch.Size([1, 10647, 1])로 바뀜

    # Treshold값 이하면 모든 속성들을 0으로 설정
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2) #torch.Size([1, 10647, 1])
    prediction = prediction * conf_mask #torch.Size([1, 10647, 85])


    # IoU를 계산하기 위하여 중심좌표 x,y,높이, 너비를 왼쪽 위 꼭지점 x,y, 오른쪽 밑 꼭지점 x,y로 변환
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    # 전체 batch에서 true detections을 모으는데 사용되는 output 이 initialize 됐는지 안됐는지 알려준다.
    write = False

    for i in range(batch_size):
        prediction = prediction[i]
        print(prediction.shape)  # torch.Size([10647, 85])

        # 가장 높은 class의 값과 idx 값
        max_conf, max_conf_idx = torch.max(prediction[:, 5:85], 1)  # torch.Size([10647])

        # torch.Size([10647, 1])
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_idx = max_conf_idx.float().unsqueeze(1)
        seq = (prediction[:, :5], max_conf, max_conf_idx)  # 5개의 Bbox 속성, 가장 높은 class, 가장 높은 score
        prediction = torch.cat(seq, 1)  # torch.Size([10647, 7])

        # confidence값보다 낮아서 0으로 된 값들 삭제
        # torch.Size([5290, 1])
        non_zero_ind = (torch.nonzero(prediction[:, 4], as_tuple=False))

        # detection이 하나도 없을때의 상황
        try:
            prediction_ = prediction[non_zero_ind.squeeze(), :].view(-1, 7)  # torch.Size([5290, 7])
        except:
            continue

        if prediction_.shape[0] == 0:
            continue

        print(prediction_[:, -1].shape)  # torch.Size([random])
        img_classes = unique(prediction_[:, -1])  # -1 index holds the class index
        print(img_classes.shape)  # 유니크한 80개의 class 추출

        # class 하나씩 NMS 적용
        '''
        NMS 작동 방법
        1. 동일한 class에 대해 confidence 순서대로 높-낮 순으로 정렬
        2. 가장 높은 confidence를 갖는 bounding box와
        IoU가 일정 threshold 이상인 다른 bounding box가 존재한다면 동일한 객체를 탐지한 것으로 판단하고 지움
        '''
        for cls in img_classes:
            ## 첫번째 단계
            # 한개의 class 에 대한 detections 가지고 옴
            cls_mask = prediction_ * (prediction_[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(cls_mask[:, -2], as_tuple=False).squeeze()

            image_pred_class = prediction_[class_mask_idx].view(-1, 7)

            # sort 한다(높-낮 순)
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  #동일한 class 내 anchor boxes 수

            ## 두번째 단계
            # 각 iteration 당 i의 인덱스를 가진 bounding box 보다 높은 index를 가진 box 중에서
            # nms_thresh보다 높은 treshold를 가지고 있는 box 제거

            for i in range(idx):
                # IoU 를 모든 박스에 대해 가지고 온다
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])

                # 비어있는 tensor
                except ValueError:
                    break

                # 제거된 box로 index error
                except IndexError:
                    break

                # 각 iteration(각 class별) 마다 i의 index를 가진 bounding box보다 높은 index를 가진 box중에서
                # nms_thresh보다 높은 threshold 가지고 있는 box 제거
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask

                # 동일한 객체 탐지한 bounding box 제거
                non_zero_ind = torch.nonzero(image_pred_class[:, 4], as_tuple= False).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)


            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(i)
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            # output이 initialize 됐는지 안됐는지, 만약 안됐다면 batch 내에 이미지에서 단 하나의 detection도 없었다는 의미이기 때문에 0 return
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


# 같은 class에 대해 여러개의 true detection이 나올 수 있기 때문에 unique 함수에서 주어진 이미지에 대해 중복되지 않은 class 값들을 가지고옴
def unique(tensor):
    tensor_np = tensor.numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res
