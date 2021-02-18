'''
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
        self.obj_scale = 1
        self.no_obj_scale = 100

    def forward(self, x, targets, labels):
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
        pred_con = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # grid cell의 좌상단의 좌표(offset)
        grid = np.arange(grid_size)
        a, b = np.meshgrid(grid, grid) #grid*grid 행렬, a와 b의 행렬 반대
        # a = [[0,1,2...12][0,1,2..12]..] (13,13)
        # b = [[0,0,0...0]...[12,12,12..12]] (13,13)

        # [1,1,grid_size, grid_size]
        cx = torch.FloatTensor(a).view([1, 1, grid_size, grid_size])
        cy = torch.FloatTensor(b).view([1, 1, grid_size, grid_size])

        # 앵커 구하기
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchor], dtype=torch.float)

        # 앵커박스의 폭&높이
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        pre_Bbox = torch.zeros_like(prediction[..., :4])
        pre_Bbox[..., 0] = tx + cx
        pre_Bbox[..., 1] = ty + cy
        pre_Bbox[..., 2] = torch.exp(tw) * anchor_w
        pre_Bbox[..., 3] = torch.exp(th) * anchor_h

        print("SCALED ANCHOR: ", scaled_anchors)
        print("SCALED ANCHOR(shape): ", scaled_anchors.shape)
        print("ANCHOR W: ", anchor_w.shape)
        print("ANCHOR H: ", anchor_h.shape)

        print("pre_Bbox:", pre_Bbox.shape) #torch.Size([1, 3, 13, 13, 4])
        pre_Bbox_=pre_Bbox.view(batch_size, -1, 4)*stride
        print("pre_Bbox:",pre_Bbox_.shape) #torch.Size([1, 507, 4])

        print("pred_con:", pred_con.shape) #torch.Size([1, 3, 13, 13])
        pred_con_ = pred_con.view(batch_size, -1, 1)
        print("pred_con:", pred_con_.shape) #torch.Size([1, 507, 1])
        print("pred_cls:", pred_cls.shape) #torch.Size([1, 3, 13, 13, 80])
        pred_cls_ = pred_cls.view(batch_size, -1, self.num_classes)
        print("pred_cls:", pred_cls_.shape) #torch.Size([1, 507, 80])

        pre_BBOX = (pre_Bbox_, pred_con_, pred_cls_) # <class: tuple>

        output = torch.cat(pre_BBOX, -1) # 마지막 차원을 기준으로 합치기
        print("output shape:", output.shape) #torch.Size([1, 507, 85])


        iou_scores, class_mask, obj_mask, no_obj_mask, xi, yi, wi, hi, clsi, confi = Target(pre_Bbox, pred_cls, targets, labels, scaled_anchors)

        # 전체 loss 계산
        # 필요한 파라미터:
        # obj_mask, no_obj_mask
        # cx, cy, w, h,
        # tx, ty, tw, th
        # pred_conf, tconf, tcls


        ## localization loss
        loss_x = nn.MSELoss()(tx[obj_mask], xi[obj_mask])
        loss_y = nn.MSELoss()(ty[obj_mask], yi[obj_mask])
        loss_w = nn.MSELoss()(tw[obj_mask], wi[obj_mask])
        loss_h = nn.MSELoss()(th[obj_mask], hi[obj_mask])
        loss_bbox = loss_x + loss_y + loss_w + loss_h

        ## confidence loss
        loss_conf_obj = nn.BCELoss()(pred_con[obj_mask], confi[obj_mask])
        loss_conf_no_obj = nn.BCELoss()(pred_con[no_obj_mask], confi[no_obj_mask])
        loss_conf = self.obj_scale * loss_conf_obj + self.no_obj_scale * loss_conf_no_obj

        ## classification loss
        loss_cls = nn.BCELoss()(pred_cls[obj_mask], clsi[obj_mask])

        loss = loss_bbox + loss_conf + loss_cls


        return output, loss

if __name__ == '__main__':
    featureMap1 = torch.randn(1, 255,13,13)
    anchors = {'scale1': [(10, 13), (16, 30), (33, 23)],
               'scale2': [(30, 61), (62, 45), (59, 119)],
               'scale3': [(116, 90), (156, 198), (373, 326)]}
    Layer=Detection(anchors['scale1'], 416, 80)
    output1 = Layer(featureMap1)


