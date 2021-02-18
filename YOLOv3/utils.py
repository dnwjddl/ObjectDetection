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

def find_intersection(set_1, set_2):
    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)  # make 0 or positive part
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)  # leave both positive parts!

def find_jaccard_overlap(set_1, set_2, eps=1e-5):
    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection + eps  # (n1, n2)

    return intersection / union  # (n1, n2)

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


# Loss를 구하기 위한 Target과의 비교
def Target(pred_boxes, pred_cls, target, labels, anchors, ignore_thres=0.5):
    # pred_boxes: [1, 3, 13, 13, 4]
    # pred_cls: [1, 3, 13, 13, 80]
    # target
    # anchor: (3,2 사이즈의 스케일 별 앵커)

    nB = pred_boxes.size(0) #1
    nA = pred_boxes.size(1) #3
    nG = pred_boxes.size(2) #grid_size
    nC = pred_cls.size(-1) #class size(80)


    ## 초기화 ##
    # [1,3,grid,grid]
    obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool) # 물체가 없으면 0, 있으면 1
    noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool) # 물체가 없으면 1, 있으면 0

    class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float) # class에 해당하지 않으면 0, 해당하면 1
    iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float) # iou score

    # ground truth 변화량
    tx = torch.zeros(nB, nA, nG, nG, dtype=torch.float)
    ty = torch.zeros(nB, nA, nG, nG, dtype=torch.float)
    tw = torch.zeros(nB, nA, nG, nG, dtype=torch.float)
    th = torch.zeros(nB, nA, nG, nG, dtype=torch.float)

    # [1,3,13,13,80]
    tcls = torch.zeros(nB, nA, nG, nG, nC, dtype=torch.float)

    ## 여기서부터 ##
    # Convert to position relative to box
    target_boxes = target[:, 1:] * nG #target의 기준이 1*1이라서, grid size 만큼 키워주기 위하여 grid size 만큼 키워줌
    gxy = target_boxes[:, :2] # ground truth의 x,y
    gwh = target_boxes[:, 2:] # ground truth의 w,h

    # Get achors with best iou
    ious = torch.stack([bbox_iou(anchor, gwh) for anchor in anchors])
    _, best_ious_idx = ious.max(0)

    # separate target values
    target_labels = labels.long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t() # 왼쪽 상단의 모서리 좌표를 할려고, 정수형으로 바꿔줌

    b = target.shape[:, 0]

    # 마스크 만들기
    obj_mask[b, best_ious_idx, gj, gi] = 1  # 물체가 있는 곳을 1
    noobj_mask[b, best_ious_idx, gj, gi] = 0 # 물체가 있는 곳을 0

    # 물체가 iou 보다 크면 물체가 있다고 판단 > noobj_mask을 0으로 만듦
    # set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # coordinates
    # ground truth의 좌표 변화량 구하기(offset)
    tx[b, best_ious_idx, gj, gi] = gx - gx.floor()
    ty[b, best_ious_idx, gj, gi] = gy - gy.floor()

    # Width and height
    tw[b, best_ious_idx, gj, gi] = torch.log(gw / anchors[best_ious_idx][:, 0] + 1e-16)
    th[b, best_ious_idx, gj, gi] = torch.log(gh / anchors[best_ious_idx][:, 1] + 1e-16)

    # One-hot encoding of label
    # 물체가 해당하는 anchor index에 1 삽입
    tcls[b, best_ious_idx, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor
    class_mask[b, best_ious_idx, gj, gi] = (pred_cls[b, best_ious_idx, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_ious_idx, gj, gi] = bbox_iou(pred_boxes[b, best_ious_idx, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()



    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names