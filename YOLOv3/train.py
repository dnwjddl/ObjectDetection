import time
import torch
import argparse
import os

from utils import load_classes
from data.datas import VOC_Dataset

from model.yolov3_anchor import YOLOv3

'''
Python의 ArgParse 모듈을 사용
--images 는 이 argument에 해당하는 값을 넣기 원할 때 먼저 나와야 하는 flag이다
dest: argument에 접근할 수 있는 이름, args.images로 해당 argument의 값에 접근할 수 있다
help: 단순히 해당 argument가 무엇을 의미하는지 알려주는 역할
default: 아무 argument를 주지 않았을 때 default의 값을 전달한다는 것을 의미
type 전달받을 argument의 자료형
'''

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--epoch", type = int, default = 100)
    parser.add_argument("--gradient_accumulation", type = int, default = 1)
    parser.add_argument("--multiscale_training", type = bool, default = True)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--num_workers", type = int, default = 8)
    #parser.add_argument("--pretrained_weights", type = str, default = 'weights/darknet53.conv.74')
    #parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "yolov3.weights", type = str)
    parser.add_argument("--image_size", type=int, default=416)
    return parser.parse_args()

    #print(args)
args = arg_parse()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
now = time.strftime('%y%m%d_%H%M%S', time.localtime(time.time()))

#Dataset
num_classes = 20 #For VOC Dataset
#classes = load_classes("data/coco.names")

#Load Model
model = YOLOv3(num_classes).to(device)

# Load Dataset
train_set = VOC_Dataset("../data/VOC2012", split='TRAIN')
'''
DataLoader: collate_fn -> 몇개의 샘플들이 배치가 되어야 하는지 지정할 수 있다
'''
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               collate_fn=train_set.collate_fn,
                                               shuffle=False, num_workers=0, pin_memory=True)
# optimizer 설정
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# learning rate scheduler 설정
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)

# # 현재 배치 손실값을 출력하는 tqdm 설정
# loss_log = tqdm.tqdm(total = 0, position = 2, bar_format = '{desc}', leave = False)
# # Train code
# for epoch in tqdm.tqdm(range(args.epoch), desc = 'Epoch'):


for epoch in range(args.epoch):
    model.train()

    for batch_idx, (images, boxes, labels , _) in enumerate(train_loader):
        step = len(train_loader) * epoch + batch_idx

        loss, outputs = model(images, boxes, labels)
        loss.backward()

        # 기울기 누적 (Accumulate gradient)
        if step % args.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

    # lr scheduler의 step을 진행
    scheduler.step()

        # 총 손실값 출력
        # loss_log.set_description_str('Loss:{:.6f}'.format(loss.item())

    '''
        # 검증 데이터셋으로 모델을 평가
        precision, recall, AP, f1, _, _, _ = evaluate(model,
                                                          path=valid_path,
                                                          iou_thres=0.5,
                                                          conf_thres=0.5,
                                                          nms_thres=0.5,
                                                          image_size=args.image_size,
                                                          batch_size=args.batch_size,
                                                          num_workers=args.num_workers,
                                                          device=device)
        
        '''
    save_dir = os.path.join('checkpoints', now)
    os.makedirs(save_dir, exist_ok= True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'yolov3_{}.pth'.format(epoch)))