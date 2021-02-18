'''
transpose() : 두개의 차원을 맞교환 가능
permute() : 모든 차원들과의 맞교환 가능
permute(1,0,2).contiguous() 같이 붙여서 사용한다

Torch 생성: torch.arange(start = 0, end, step = 1, out =None, dtype)
torch.repeat : x.repeat(4,2) -> 4*6 (dim = 0으로 4, dim= 1으로 2 만큼 확대)
'''

import torch
import torch.nn as nn
import numpy as np
from utils import Target

class Detection(nn.Module):
    def __init__(self, anchors, image_size: int, num_classes: int):
        super(Detection, self).__init__()
        self.anchor = anchors
        self.num_anchors = len(anchors) #앵커 수
        self.num_classes = num_classes #class 수
        self.img_size = image_size #이미지 사이즈 (416)

    def forward(self, x, targets):
        # feature map의 크기: [1,255,13,13] 이라면
        # batch_size : 1
        # grid_size : 13
        # stride : 255 / 13 = 32

        batch_size = x.size(0)
        grid_size = x.size(2)
        stride = self.img_size / grid_size


        # 출력값 형태 변환
        # [1, 3, 85, 13, 13]
        prediction = x.view(batch_size, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
        # [1, 3, 13, 13, 85]
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()

        tx = torch.sigmoid(prediction[..., 0])  # Center x
        ty = torch.sigmoid(prediction[..., 1])  # Center y
        tw = prediction[..., 2]  # Width
        th = prediction[..., 3]  # Height
        anchor_con = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        anchor_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # grid cell의 좌상단의 좌표(offset)
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid) #grid*grid 행렬, a와 b의 행렬 반대
        # [1,1,grid_size, grid_size]
        cx = torch.FloatTensor(a).view([1, 1, grid_size, grid_size])
        cy = torch.FloatTensor(b).view([1, 1, grid_size, grid_size])

        # 앵커 구하기
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchor], dtype=torch.float)
        print("scaled_anchors:", scaled_anchors)

        # 앵커박스의 폭&높이
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        Bbox = torch.zeros_like(prediction[..., :4])
        Bbox[..., 0] = tx + cx
        Bbox[..., 1] = ty + cy
        Bbox[..., 2] = torch.exp(tw) * anchor_w
        Bbox[..., 3] = torch.exp(th) * anchor_h


        # 3 * grid * grid  합치기
        print("Bbox:", Bbox.shape) #torch.Size([1, 3, 13, 13, 4])
        Bbox_=Bbox.view(batch_size, -1, 4) *stride
        print("Bbox_:",Bbox_.shape) #torch.Size([1, 507, 4])


        print("anchor_con:", anchor_con.shape) #torch.Size([1, 3, 13, 13])
        anchor_con_ = anchor_con.view(batch_size, -1, 1)
        print("anchor_con_:", anchor_con_.shape) #torch.Size([1, 507, 1])


        print("anchor_cls:", anchor_cls.shape) #torch.Size([1, 3, 13, 13, 80])
        anchor_cls_ = anchor_cls.view(batch_size, -1, self.num_classes)
        print("anchor_cls_:", anchor_cls_.shape) #torch.Size([1, 507, 80])


        # 합치기
        BBOX = (Bbox_, anchor_con_, anchor_cls_) # <class: tuple>

        output = torch.cat(BBOX, -1) # 마지막 차원을 기준으로 합치기
        print("output shape:", output.shape) #torch.Size([1, 507, 85])

        if targets is None:
             return output

        iou_scores, class_mask, obj_mask, no_obj_mask, tx, ty, tw, th, tcls, tconf = Target(pred_boxes=Bbox, pred_cls=anchor_cls, target=targets, anchors=scaled_anchors)


        print("TEST+++++++++++++++++++++++++++")
        nB = Bbox.size(0)  # 1
        nA = Bbox.size(1)  # 3
        nG = anchor_cls.size(2)  # grid_size
        nC = Bbox.size(-1)  # class size(80)

        # tensor
        # [1,3,grid,grid]
        print(nB, nA, nG, nC)

        obj_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.bool)  # 물체가 없으면 0, 있으면 1
        print("obj_mask", obj_mask)
        print("obj_mask.size", obj_mask.shape)
        noobj_mask = torch.ones(nB, nA, nG, nG, dtype=torch.bool)  # 물체가 없으면 1, 있으면 0
        print("noobj_mask", noobj_mask)
        print("noobj_mask.size", noobj_mask.shape)

        class_mask = torch.zeros(nB, nA, nG, nG, dtype=torch.float)  # class에 해당하지 않으면 0, 해당하면 1
        iou_scores = torch.zeros(nB, nA, nG, nG, dtype=torch.float)  # iou score
        print("class_mask", class_mask)
        print("class_mask.size", class_mask.shape)
        print("iou_scores", iou_scores)
        print("iou_scores.size", iou_scores.shape)

        return output