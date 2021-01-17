import torch
import torch.nn as nn

class Detection(nn.Module):
    def __init__(self, anchors, image_size:int, num_classes: int):
        super(Detection, self).__init__()
        self.anchor = anchors
        self.num_anchors = len(anchors) #앵커 수
        self.num_classes = num_classes #class 수
        self.img_size = image_size #이미지 사이즈 (416)

    def forward(self, x):
        # feature map의 크기: [1,255,13,13] 이라면
        # num_batches : 1
        # grid_size : 13

        num_batches = x.size(0)
        grid_size = x.size(2)

        # 출력값 형태 변환
        '''
        transpose() : 두개의 차원을 맞교환 가능
        permute() : 모든 차원들과의 맞교환 가능
        permute(1,0,2).contiguous() 같이 붙여서 사용한다
        - view()를 쓰면 알아서 순서 고려해서 바꿔주지만 같이 쓰는 것이 좋음
        '''
        # [1, 3, 85, 13, 13]
        # [1, 3, 13, 13, 85]
        prediction = (
            x.view(num_batches, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
                .permute(0, 1, 3, 4, 2).contiguous()
        )

        # Get outputs
        # 85에서 5 + 80 가지고 오기

        '''
        질문: 시그모이드를 왜 해줌?
        '''
        tx = torch.sigmoid(prediction[..., 0])  # Center x
        ty = torch.sigmoid(prediction[..., 1])  # Center y
        tw = prediction[..., 2]  # Width
        th = prediction[..., 3]  # Height
        anchor_con = torch.sigmoid(prediction[..., 4])  # Object confidence (objectness)
        anchor_cls = torch.sigmoid(prediction[..., 5:])  # Class prediction

        # Calculate offsets for each grid

        ## 32, 16, 8 stride 순서
        stride = self.img_size / grid_size

        # Torch 생성: torch.arange(start = 0, end, step = 1, out =None, dtype)
        # torch.repeat : x.repeat(4,2) -> 4*6 (dim = 0으로 4, dim= 1으로 2 만큼 확대)

        # grid cell의 좌상단의 좌표(offset)
        # [1,1,grid_size, grid_size]
        cx = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).view([1, 1, grid_size, grid_size])

        # .t() -> 행렬의 행과 열 바꾸기
        cy = torch.arange(grid_size, dtype=torch.float).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size])

        # 앵커 구하기
        scaled_anchors = torch.as_tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchor], dtype=torch.float)
        # 앵커박스의 폭&높이
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        Bbox = torch.zeros_like(prediction[..., :4])
        Bbox[..., 0] = tx + cx
        Bbox[..., 1] = ty + cy
        Bbox[..., 2] = torch.exp(tw) * anchor_w
        Bbox[..., 3] = torch.exp(th) * anchor_h

        BBOX = (Bbox.view(num_batches, -1, 4) * stride,
                anchor_con.view(num_batches, -1, 1),
                anchor_cls.view(num_batches, -1, self.num_classes))

        output = torch.cat(BBOX, -1) # 마지막 차원을 기준으로 합치기

        return output
