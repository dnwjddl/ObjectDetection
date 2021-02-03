### ```yolo_cfg```폴더
- cfg 파일과 weight을 사용하여 물체 감지
  - ```yolo.py```: 사진 객체 감지
  - ```yolo_video.py``` : 웹캠을 사용하여 실시간 객체 감지

## 결과값 

![image](https://user-images.githubusercontent.com/72767245/104844285-c12a7280-5912-11eb-986e-5b49a645cc3d.png)

### 최종결과값에 대한 설명
```torch.Size([1, 10647, 85])```  
세개 다른 scale 의 output tensor 합친 값 + 80(class 갯수) + 5(위치+confidence값)  

```torch.Size([random, 8])```  

- 8개의 속성
  - 1 = batch 에서 이미지 인덱스
  - 2~5 = 꼭지점의 왼위, 오아 좌표
  - 6 = Objectness 점수
  - 7 = Maximum confidence를 가진 class 점수
  - 8 = 그 class의 index 값


![image](https://user-images.githubusercontent.com/72767245/104838191-a0efb900-58fc-11eb-9837-0612d74e3946.png)

- ```Darknet53.py``` : DarkNet-53 Layer 제작
- ```[TEST] yolov3_architecture.py``` : Layer Model 제작

#### cfg 파일 (Configuration file)

- cfg 파일은 블록단위로 네트워크의 Layout을 나타냄

저자가 제공하는 공식 cfg 파일 사용  

```python
cd cfg
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
```

**Configuration** 파일을 열면

```c
[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear
```
저자의 cfg 파일을 토대로 ```yolov3_architecture.py``` 작성

```c
[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .5
truth_thresh = 1
random=1
```

```c
[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=16
width= 320
height = 320
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1
```
```python
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
```

![image](https://user-images.githubusercontent.com/72767245/104836940-af39d700-58f4-11eb-8474-6c645968c2ee.png)
